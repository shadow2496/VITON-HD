## VITON-HD &mdash; Official PyTorch Implementation

![Teaser image](./assets/teaser.png)

> **VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization**<br>
> [Seunghwan Choi](https://github.com/shadow2496)\*<sup>1</sup>, [Sunghyun Park](https://psh01087.github.io)\*<sup>1</sup>, [Minsoo Lee](https://github.com/Minsoo2022)\*<sup>1</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo)<sup>1</sup><br>
> <sup>1</sup>KAIST<br>
> In CVPR 2021. (* indicates equal contribution)

> Paper: https://arxiv.org/abs/2103.16874<br>
> Project page: https://psh01087.github.io/VITON-HD

> **Abstract:** *The task of image-based virtual try-on aims to transfer a target clothing item onto the corresponding region of a person, which is commonly tackled by fitting the item to the desired body part and fusing the warped item with the person. While an increasing number of studies have been conducted, the resolution of synthesized images is still limited to low (e.g., 256x192), which acts as the critical limitation against satisfying online consumers. We argue that the limitation stems from several challenges: as the resolution increases, the artifacts in the misaligned areas between the warped clothes and the desired clothing regions become noticeable in the final results; the architectures used in existing methods have low performance in generating high-quality body parts and maintaining the texture sharpness of the clothes. To address the challenges, we propose a novel virtual try-on method called VITON-HD that successfully synthesizes 1024x768 virtual try-on images. Specifically, we first prepare the segmentation map to guide our virtual try-on synthesis, and then roughly fit the target clothing item to a given person's body. Next, we propose ALIgnment-Aware Segment (ALIAS) normalization and ALIAS generator to handle the misaligned areas and preserve the details of 1024x768 inputs. Through rigorous comparison with existing methods, we demonstrate that VITON-HD highly surpasses the baselines in terms of synthesized image quality both qualitatively and quantitatively.*

## Installation

Clone this repository:

```
git clone https://github.com/shadow2496/VITON-HD.git
cd ./VITON-HD/
```

Install PyTorch and other dependencies:

```
conda create -y -n [ENV] python=3.8
conda activate [ENV]
conda install -y pytorch=[>=1.6.0] torchvision cudatoolkit=[>=9.2] -c pytorch
pip install opencv-python torchgeometry
```

## Pre-trained networks

We cannot share the training code or the collected dataset due to the commercial issue. Instead, we provide pre-trained networks and sample images from the test dataset. Please download `*.pkl` and dataset-related files from the [VITON-HD Google Drive folder](https://drive.google.com/drive/folders/0B8kXrnobEVh9fnJHX3lCZzEtd20yUVAtTk5HdWk2OVV0RGl6YXc0NWhMOTlvb1FKX3Z1OUk?resourcekey=0-OIXHrDwCX8ChjypUbJo4fQ&usp=sharing) and unzip `*.zip` files. `test.py` assumes that the downloaded files are placed in `./checkpoints/` and `./datasets/` directories.

## Testing

To generate virtual try-on images, run:

```
CUDA_VISIBLE_DEVICES=[GPU_ID] python test.py --name [NAME]
```

The results are saved in the `./results/` directory. You can change the location by specifying the `--save_dir` argument. To synthesize virtual try-on images with different pairs of a person and a clothing item, edit `./datasets/test_pairs.txt` and run the same command.

## License

All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license by [NeStyle Inc](http://www.nestyle.ai). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{choi2021viton,
  title={VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization},
  author={Choi, Seunghwan and Park, Sunghyun and Lee, Minsoo and Choo, Jaegul},
  booktitle={Proc. of the IEEE conference on computer vision and pattern recognition (CVPR)},
  year={2021}
}
```
