## Welcome to SCGAN Homepage

An official PyTorch Implementation of paper "Spatially Constrained GAN for Face and Fashion Synthesis" in FG2021.

By [Songyao Jiang](https://www.songyaojiang.com/), [Hongfu Liu](http://hongfuliu.com/), [Yue Wu](http://wuyuebupt.github.io/) and [Yun Fu](http://www1.ece.neu.edu/~yunfu/).

[Smile Lab @ Northeastern University](https://web.northeastern.edu/smilelab/)

## Data Preparation

1. Download [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). We used their aligned&cropped version. Preprocessed segmentation data for CelebA is provided at [GoogleDrive](https://drive.google.com/file/d/1K496cZAlssIvrbW8ygzivYobWvQuAaGM/view?usp=sharing).

2. Download [DeepFashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). We used their [Fashion Synthesis Subset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html)

## Train
```
bash scripts/train_celeba_command.sh
bash scripts/train_fashion_command.sh
```
## Test
```
bash scripts/test_celeba_command.sh
bash scripts/test_fashion_command.sh
```
## Citation
If you find this repo useful in your research, please consider citing 
```
@inproceedings{jiang2021spatially,
  title={Spatially Constrained GAN for Face and Fashion Synthesis},
  author={Jiang, Songyao and Tao, Zhiqiang and Fu, Yun},
  booktitle={2021 16th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2021)},
  year={2019},
  organization={IEEE}
}

@inproceedings{jiang2019segmentation,
  title={Segmentation guided image-to-image translation with adversarial networks},
  author={Jiang, Songyao and Tao, Zhiqiang and Fu, Yun},
  booktitle={2019 14th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2019)},
  pages={1--7},
  year={2019},
  organization={IEEE}
}
```

<!-- ```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
``` -->