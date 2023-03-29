# ELIC
This repository is an unofficial PyTorch implementation variants of ELIC: efficient Learned Image Compression withUnevenly Grouped Space-Channel Contextual Adaptive Coding [(CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html).

>  This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [TinyLIC](https://github.com/lumingzzz/TinyLIC).


## Training

```
CUDA_VISIBLE_DEVICES=0 python train.py -m elic -d /path/to/Flicker2W -q 2  --epochs 500 -lr 1e-4 --batch-size 8 --cuda --save
```
By adjusting the quality parameters ```q```(1-9) in the above command line, you can make rate distortion tradeoffs. I have mapped the quality parameters to the lambda values following CompressAI.


## Testing

```
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model checkpoint /path/to/Kodak -a elic -p /path/to/pretrained/elic/2/checkpoint_best_loss.pth.tar --cuda
```



## About

I replaced the checkerboard context model in the original ELIC with Multistage Context Model (MCM) in TinyLIC to further enhance spatial context information. My reimplementation for other modules is as consistent as possible with the description in the ELIC.


## Contact

The codes have been released. If you have any question, please contact me via wanghr827@whu.edu.cn.
