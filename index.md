## Welcome to SCGAN Homepage

An official PyTorch implementation of paper "Spatially Constrained GAN for Face and Fashion Synthesis" in FG2021.

By [Songyao Jiang](https://www.songyaojiang.com/), [Hongfu Liu](http://hongfuliu.com/), [Yue Wu](http://wuyuebupt.github.io/) and [Yun Fu](http://www1.ece.neu.edu/~yunfu/).

[Smile Lab @ Northeastern University](https://web.northeastern.edu/smilelab/)

## Problem Definition
### Goal
Decouple the image synthesis task into three dimensions (i.e., spatial, attribute and latent dimensions), control the spatial and attribute-level contents, and randomize the other unregulated contents. Our goal can be described as finding the mapping 

<img src="https://render.githubusercontent.com/render/math?math=G\left(z,c,s\right)\rightarrow y">

where <img src="https://render.githubusercontent.com/render/math?math=G(\cdot,\cdot,\cdot)"> is the generating function, <img src="https://render.githubusercontent.com/render/math?math=z"> is the latent vector of size <img src="https://render.githubusercontent.com/render/math?math=($1 \times n_z$)">, and <img src="https://render.githubusercontent.com/render/math?math=y"> is the conditionally generated image which complies with the target conditions <img src="https://render.githubusercontent.com/render/math?math=c"> and <img src="https://render.githubusercontent.com/render/math?math=s">. 

### Motivations
- Face and fashion synthesis are inherently one-to-many mapping from semantic segmentations to real images.
### Key Contributions
- SCGAN decouples the face and fashion synthesis task into three dimensions (spatial, attribute, and latent). 
- A particularly designed generator extracts spatial information from segmentation, utilizes variations in random latent vectors and applies specified attributes. A segmentor network guides the generator with spatial constraints and improves model convergence.
- Extensive experiments on the CelebA and DeepFashion datasets demonstrate the effectiveness of SCGAN.

## Method

### SCGAN Framework
Our proposed SCGAN consists of three networks shown below, which are a generator network G, a discriminator network D, and a segmentor network S. 
[<img src="img/framework.png" width = "600">](img/framework.png)
- We utilize a generator network G to match our desired mapping function <img src="https://render.githubusercontent.com/render/math?math=G\left(z,c,s\right)\rightarrow y">. generator takes three inputs which are a latent code z, an attribute label c, and a target segmentation map s. As shown in the above figure, these inputs are fed into the generator step by step in orders. This particular design of G decides the spatial configuration of the synthesized image according to the spatial constraints extracted from s. Then G forms the basic structure (__e.g.__, background, ambient lighting) of the generated image using the information coded in z. After that, G generates the attribute components specified by c.
- We employ a discriminator network D which forms a GAN framework with G. An auxiliary classifier is embedded in D to do a multi-class classification which provides attribute-level and domain-specific information back to G.
- We propose a segmentor network S to provide spatial constraints in conditional image generation. S takes either real or generated image data as input and outputs the probabilities of pixel-wise semantic segmentation results

### Objective Functions
- Adversarial Loss  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{adv}={L}_{adv}^{real}+{L}_{adv}^{fake}+{L}_{gp}">,    
- Classification Loss  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{cls}^{real}=\mathbb{E}_{x,c}\left[A_c(c,D_{c}(x))\right]">,   
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{cls}^{fake}=\mathbb{E}_{z,c,s}\left[A_c(c,D_{c}(G(z,c,s)))\right]">,  
- Segmentation Loss  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{seg}^{real}=\mathbb{E}_{x,s}[A_s(s, S(x)]">,  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{seg}^{fake}=\mathbb{E}_{z,c,s}\left[A_s(s, S(G(z, c, s)))\right]">,  
- Overall Objectives:  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{S}=\mathcal{L}_{seg}^{real}">,  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{D}=-\mathcal{L}_{adv}+\lambda_{cls}\mathcal{L}_{cls}^{real}">,  
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}=\mathcal{L}_{adv}^{fake}+\lambda_{cls}\mathcal{L}_{cls}^{fake}+\lambda_{seg}\mathcal{L}_{seg}^{fake}">,


## Experiment
We verify the effectiveness of SCGAN on a face dataset **CelebA** and a fashion dataset **DeepFashion**. We show both visual and quantitative results compared with four representative methods. 
### Qualitative Results
[<img src="img/compare_celeba.png" width = "600">](img/compare_celeba.png)

[<img src="img/compare_deepfashion.png" width = "600">](img/compare_deepfashion.png)

### Quantitative Evaluation
<img src="img/quantitative.PNG" width = "400">

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