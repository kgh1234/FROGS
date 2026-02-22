# FROGS : Fast Representations with Object-centric Gaussian Splatting



## Environment Setup
We follow the standard **3D Gaussian Splatting (3DGS)** environment.

### 1. Clone the Repository

```bash
git clone https://github.com/kgh1234/FROGS.git --recursive
```


### (2) Environment Configuration

For convenience, we provide a pre-built Docker environment!

``` bash
docker pull mobuk/frogs:stable
```

This Docker image does not include the source code.
Please use the mount option to link the cloned GitHub repository.

``` bash
docker run -it -v ${PWD}:/workspace --gpus all --shm-size=16g --name frogs mobuk/frogs:stable /bin/bash
```


### (3) Dataset Preparation

We mainly use two datasets: Mip-NeRF 360 and LERF-Mask.

Official Download link
- MipNeRF360 : 
- LERF-Mask : 
- Tanks & Temples : 

In this paper, additional mask annotations are required.
Please download them from the link below.
If the link does not work, contact now0104@knu.ac.kr .

Download link: Coming soon


After downloading, organize the datasets as follows:

```
FROGS
├── efficientvit_sam
├── environment
├── gaussian_splatting
├── mip-splatting
│
masked_datasets
├── lerf_mask
│   ├── figurines_15
│   ├── figurines_33
│   ├── figurines_36
│   ├── ...
│
├── mipnerf
│   ├── bicycle
│   ├── bonsai
│   ├── ...
│
├── tanks_temples
│   ├── Barn
│   ├── Caterpillar
│   ├── ...

```


### (3) Run

Edit Masked_3dgs_train_all.sh to select the dataset:

``` bash
# Masked_3dgs_train_all.sh

SCENE_NAME="mipnerf" # or lerf_mask

```

Run the script from the following directory:
``` bash
# PWD: FROGS/gaussian_splatting
bash Masked_3dgs_train_all.sh
```




## Author Info.