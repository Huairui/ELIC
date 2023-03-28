import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MultistageMaskedConv2d
from compressai.layers import (
    AttentionBlock,
    ResidualBottleneckBlock,
    conv3x3,
    subpel_conv3x3,
)
from .utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer,quantize_ste, Demultiplexerv2, Multiplexerv2


class UniversalQuant(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, x):
        b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
                                       * (2**b), 0.5 * torch.ones(x.size()) * (2**b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @ staticmethod
    def backward(ctx, g):
        return g


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels, init_weights=None):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(
            entropy_bottleneck_channels)

        if init_weights is not None:
            warnings.warn(
                "init_weights was removed as it was never functional",
                DeprecationWarning,
            )

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
class ResBottleneckGroup(nn.Sequential):
            def __init__(
                    self, in_channels: int = 192, out_channels: int = 192):
                super().__init__(
                    ResidualBottleneckBlock(in_channels, out_channels),
                    ResidualBottleneckBlock(out_channels, out_channels),
                    ResidualBottleneckBlock(out_channels, out_channels),
                )

class ELIC(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()

        

        class Encoder(nn.Sequential):
            def __init__(
                    self, in_channels: int = 3, mid_channels: int = 192, out_channels: int = 320):
                super().__init__(
                    conv(in_channels, mid_channels, stride=2),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    conv(mid_channels, mid_channels, stride=2),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    AttentionBlock(mid_channels),
                    conv(mid_channels, mid_channels, stride=2),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    conv(mid_channels, out_channels, stride=2),
                    AttentionBlock(out_channels),
                )

        class Decoder(nn.Sequential):
            def __init__(
                    self, in_channels: int = 320, mid_channels: int = 192, out_channels: int = 3):
                super().__init__(
                    AttentionBlock(in_channels),
                    subpel_conv3x3(in_channels, mid_channels, 2),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    subpel_conv3x3(mid_channels, mid_channels, 2),
                    AttentionBlock(mid_channels),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    subpel_conv3x3(mid_channels, mid_channels, 2),
                    ResBottleneckGroup(mid_channels, mid_channels),
                    subpel_conv3x3(mid_channels, out_channels, 2),
                )

        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_channels: int = 320, mid_channels: int = 320, out_channels: int = 320
            ):
                super().__init__(
                    conv3x3(in_channels, mid_channels),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(mid_channels, mid_channels),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(mid_channels, mid_channels, stride=2),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(mid_channels, mid_channels),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(mid_channels, out_channels, stride=2),
                )

        class HyperDecoder(nn.Sequential):
            def __init__(
                    self, channels: int = 320):
                super().__init__(
                    conv3x3(channels, channels),
                    nn.LeakyReLU(inplace=True),
                    subpel_conv3x3(channels, channels, 2),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(channels, channels * 3 // 2),
                    nn.LeakyReLU(inplace=True),
                    subpel_conv3x3(channels * 3 // 2, channels * 3 // 2, 2),
                    nn.LeakyReLU(inplace=True),
                    conv3x3(channels * 3 // 2, channels * 2)
                )

        class Hyperprior(CompressionModel):
            def __init__(self, channels: int = 320, mid_channels: int = 320):
                super().__init__(entropy_bottleneck_channels=mid_channels)
                self.num_slices_list = [16, 16, 32, 64, 192]
                self.slice_end_list = [16, 32, 64, 128]
                entropy_parameters_channel_list = [
                    [mid_channels*2 + 16*2 + 0, 16*32,  16*8,  16*2],
                    [mid_channels*2 + 16*2 + 16, 16*32,  16*8,  16*2],
                    [mid_channels*2 + 32*2 + 32, 32*16,  32*8,  32*2],
                    [mid_channels*2 + 64*2 + 64, 64 * 8,  64*4,  64*2],
                    [mid_channels*2 + 192*2 + 128, 192*4, 192*4, 192*2],
                ]
                self.num_slices = 5

                self.hyper_encoder = HyperEncoder(
                    channels, mid_channels, channels)
                self.hyper_decoder = HyperDecoder(mid_channels)

                self.context_prediction_1 = nn.ModuleList(
                    MultistageMaskedConv2d(
                        num_slices, num_slices * 2, kernel_size=3, padding=1, stride=1, mask_type='A')
                    for num_slices in self.num_slices_list)

                self.context_prediction_2 = nn.ModuleList(
                    MultistageMaskedConv2d(
                        num_slices, num_slices * 2, kernel_size=3, padding=1, stride=1, mask_type='B')
                    for num_slices in self.num_slices_list)
                self.context_prediction_3 = nn.ModuleList(
                    MultistageMaskedConv2d(
                        num_slices, num_slices * 2, kernel_size=3, padding=1, stride=1, mask_type='C')
                    for num_slices in self.num_slices_list)

                self.gaussian_conditional = GaussianConditional(None)
                self.entropy_parameters = nn.ModuleList(
                    nn.Sequential(
                        conv(ep_channel[0], ep_channel[1], 1, 1),
                        nn.GELU(),
                        conv(ep_channel[1], ep_channel[2], 1, 1),
                        nn.GELU(),
                        conv(ep_channel[2], ep_channel[3], 1, 1)
                    ) for ep_channel in entropy_parameters_channel_list)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                params = self.hyper_decoder(z_hat)
                b, c, latent_h, latent_w = params.size()
                y_slices = torch.tensor_split(y, self.slice_end_list, 1)
                y_hat_slices = []
                y_likelihood = []

                for slice_index, y_slice in enumerate(y_slices):
                    support_slices = y_hat_slices
                    support = torch.cat([params] + support_slices, dim=1)

                    ctx_params_0 = torch.zeros(
                        b, self.num_slices_list[slice_index]*2, latent_h, latent_w).to(z_hat.device)
                    ctx_params=ctx_params_0
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                    y_1 = y_hat_slice.clone()
                    y_1[:, :, 0::2, 1::2] = 0
                    y_1[:, :, 1::2, :] = 0
                    ctx_params_1 = self.context_prediction_1[slice_index](y_1)
                    ctx_params_1[:, :, 0::2, :] = 0
                    ctx_params_1[:, :, 1::2, 0::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                    y_2 = y_hat_slice.clone()
                    y_2[:, :, 0::2, 1::2] = 0
                    y_2[:, :, 1::2, 0::2] = 0
                    ctx_params_2 = self.context_prediction_2[slice_index](y_2)
                    ctx_params_2[:, :, 0::2, 0::2] = 0
                    ctx_params_2[:, :, 1::2, :] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                    y_3 = y_hat_slice.clone()
                    y_3[:, :, 1::2, 0::2] = 0
                    ctx_params_3 = self.context_prediction_3[slice_index](y_3)
                    ctx_params_3[:, :, 0::2, :] = 0
                    ctx_params_3[:, :, 1::2, 1::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2+ctx_params_3
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                    y_hat_slices.append(y_hat_slice)

                    _, y_slice_likelihood = self.gaussian_conditional(
                        y_slice, scales_hat, means=means_hat)
                    y_likelihood.append(y_slice_likelihood)

                y_hat = torch.cat(y_hat_slices, dim=1)
                y_likelihoods = torch.cat(y_likelihood, dim=1)
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)
                z_strings = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(
                    z_strings, z.size()[-2:])
                params = self.hyper_decoder(z_hat)

                y_slices = torch.tensor_split(y, self.slice_end_list, 1)
                
                b, c, latent_h, latent_w = params.size()

                cdf = self.gaussian_conditional.quantized_cdf.tolist()
                cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
                offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
                encoder = BufferedRansEncoder()
                symbols_list = []
                indexes_list = []
                y_hat_slices = []
                y_strings = []

                for slice_index, y_slice in enumerate(y_slices):
                    y_slice_0, y_slice_1, y_slice_2, y_slice_3 = Demultiplexerv2(y_slice)
                    support_slices = y_hat_slices
                    support = torch.cat([params] + support_slices, dim=1)
                    ctx_params_0 = torch.zeros(
                        b, self.num_slices_list[slice_index]*2, latent_h, latent_w).to(z_hat.device)
                    ctx_params=ctx_params_0
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                    means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                    index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                    y_q_slice_0 = self.gaussian_conditional.quantize(y_slice_0, "symbols", means_hat_0)
                    y_hat_slice_0 = y_q_slice_0 + means_hat_0

                    symbols_list.extend(y_q_slice_0.reshape(-1).tolist())
                    indexes_list.extend(index_0.reshape(-1).tolist())


                    y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))

                    ctx_params_1 = self.context_prediction_1[slice_index](y_hat_slice)
                    ctx_params_1[:, :, 0::2, :] = 0
                    ctx_params_1[:, :, 1::2, 0::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                    _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                    index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                    y_q_slice_1 = self.gaussian_conditional.quantize(y_slice_1, "symbols", means_hat_1)
                    y_hat_slice_1 = y_q_slice_1 + means_hat_1
                    symbols_list.extend(y_q_slice_1.reshape(-1).tolist())
                    indexes_list.extend(index_1.reshape(-1).tolist())

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))

                    ctx_params_2 = self.context_prediction_2[slice_index](y_hat_slice)
                    ctx_params_2[:, :, 0::2, 0::2] = 0
                    ctx_params_2[:, :, 1::2, :] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                    _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                    index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                    y_q_slice_2 = self.gaussian_conditional.quantize(y_slice_2, "symbols", means_hat_2)
                    y_hat_slice_2 = y_q_slice_2 + means_hat_2

                    symbols_list.extend(y_q_slice_2.reshape(-1).tolist())
                    indexes_list.extend(index_2.reshape(-1).tolist())

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            y_hat_slice_2, torch.zeros_like(y_hat_slice_0))
                    ctx_params_3 = self.context_prediction_3[slice_index](y_hat_slice)
                    ctx_params_3[:, :, 0::2, :] = 0
                    ctx_params_3[:, :, 1::2, 1::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2+ctx_params_3
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                    _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                    index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                    y_q_slice_3 = self.gaussian_conditional.quantize(y_slice_3, "symbols", means_hat_3)
                    y_hat_slice_3 = y_q_slice_3 + means_hat_3

                    symbols_list.extend(y_q_slice_3.reshape(-1).tolist())
                    indexes_list.extend(index_3.reshape(-1).tolist())

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                    y_hat_slices.append(y_hat_slice)
                
                encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
                
                
                y_string = encoder.flush()
                y_strings.append(y_string)
                return {
                    "strings": [y_strings, z_strings],
                    "shape": z.size()[-2:],
                }

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
                params = self.hyper_decoder(z_hat)
                latent_h, latent_w = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

                y_hat_slices = []
                y_string = strings[0][0]

                cdf = self.gaussian_conditional.quantized_cdf.tolist()
                cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
                offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
                decoder = RansDecoder()
                decoder.set_stream(y_string)

                for slice_index in range(self.num_slices):
                    support_slices = y_hat_slices
                    support = torch.cat([params] + support_slices, dim=1)
                    # Stage 0:
                    ctx_params_0 = torch.zeros(
                        z_hat.shape[0], self.num_slices_list[slice_index]*2, latent_h, latent_w).to(z_hat.device)
                    ctx_params=ctx_params_0
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1))

                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                    means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                    index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                    rv = decoder.decode_stream(index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                    rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                    y_hat_slice_0 = self.gaussian_conditional.dequantize(rv, means_hat_0)

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                                torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))

                    # Stage 1:
                    ctx_params_1 = self.context_prediction_1[slice_index](y_hat_slice)
                    ctx_params_1[:, :, 0::2, :] = 0
                    ctx_params_1[:, :, 1::2, 0::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1))
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                    _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                    index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                    rv = decoder.decode_stream(index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                    rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                    y_hat_slice_1 = self.gaussian_conditional.dequantize(rv, means_hat_1)

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                                torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))

                    # Stage 2:
                    ctx_params_2 = self.context_prediction_2[slice_index](y_hat_slice)
                    ctx_params_2[:, :, 0::2, 0::2] = 0
                    ctx_params_2[:, :, 1::2, :] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                    _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                    index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                    rv = decoder.decode_stream(index_2.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                    rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                    y_hat_slice_2 = self.gaussian_conditional.dequantize(rv, means_hat_2)
                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                                y_hat_slice_2, torch.zeros_like(y_hat_slice_0))


                    # Stage 3:
                    ctx_params_3 = self.context_prediction_3[slice_index](y_hat_slice)
                    ctx_params_3[:, :, 0::2, :] = 0
                    ctx_params_3[:, :, 1::2, 1::2] = 0
                    ctx_params=ctx_params_0+ctx_params_1+ctx_params_2+ctx_params_3
                    gaussian_params = self.entropy_parameters[slice_index](
                        torch.cat((support, ctx_params), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)
                    _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                    _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                    index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                    rv = decoder.decode_stream(index_3.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                    rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                    y_hat_slice_3 = self.gaussian_conditional.dequantize(rv, means_hat_3)

                    y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                    y_hat_slices.append(y_hat_slice)

                y_hat = torch.cat(y_hat_slices, dim=1)
                return y_hat

        self.g_a = Encoder(3, N, M)
        self.g_s = Decoder(M, N, 3)
        self.hyperprior = Hyperprior(M, M)

    def forward(self, x):
        y = self.g_a(x)
        y_hat, likelihoods = self.hyperprior(y)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": likelihoods['y'], "z": likelihoods['z']},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.g_a_6.weight"].size(0)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_dict = self.hyperprior.compress(y)
        y_hat = self.hyperprior.decompress(y_dict["strings"], y_dict["shape"])
        # decode the motion information
        x_hat = self.g_s(y_hat)
        return {"rec_img": x_hat, "strings": y_dict["strings"], "shape": y_dict["shape"]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        y_hat = self.hyperprior.decompress(strings, shape)
        # decode the motion information
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = 0
        for n, m in self.named_modules():
            if isinstance(m, CompressionModel):
                aux_loss += m.aux_loss()
        return aux_loss

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.hyperprior.gaussian_conditional.update_scale_table(
            scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(
            force=force)
        return updated
