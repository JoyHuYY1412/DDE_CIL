# CVPR2021 Incremental Learning
[[Paper]](https://arxiv.org/abs/2103.01737) 

This repository is for the paper "Distilling Causal Effect of Data in Class-Incremental Learning".
<div align="center">
  <img width="70%", src="https://github.com/JoyHuYY1412/DDE_CIL/blob/main/illu.jpg"/>
</div><br/>


# Instructions
1. Dependencies
	- Python 3.6 (Anaconda3 Recommended)
	- Pytorch 0.4.0
	- torchvision 0.2.1 
	- numpy 1.18.1

2. Getting Started 
	- the data for CIFAR100 and ImageNet are put in `cifar100-class-incremental/data` and `imagenet-class-incremental/data`, or you can make soft links to the directories which include the corresponding data
	- make soft links for `utils_incremental` folder under `cifar100-class-incremental` and `imagenet-class-incremental`
	- make folders `logs`, `results` and `checkpoint` under `cifar100-class-incremental` and `imagenet-class-incremental`
	- see `cifar100-class-incremental/run.sh` for the experiments on CIFAR100
	- see `imagenet-class-incremental/run.sh` for the experiments on ImageNet-Subset
	- see `imagenet-class-incremental/run_all.sh` for the experiments on ImageNet-Full

# Citation
Please cite the following paper if you find this useful in your research:
```
@InProceedings{Hu_20121_CVPR,
author = {Hu, Xinting and Tang, Kaihua and Miao, Chunyan and Hua, Xian-Sheng and Zhang, Hanwang},
title = {Distilling Causal Effect of Data in Class-Incremental Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```
