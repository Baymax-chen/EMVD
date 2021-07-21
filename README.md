# EMVD
Efficient Multi-Stage Video Denoising With Recurrent Spatio-Temporal Fusion.

EMVD is an efficient video denoising method which recursively exploit the spatio temporal correlation inherently present in natural videos through multiple cascading processing stages applied in a recurrent fashion, namely temporal fusion, spatial denoising, and spatio-temporal refinement.

# Overview
This repo. is an ***unofficial*** version od EMVD mentioned by **Matteo Maggioni, Yibin Huang, Cheng Li, Shuai Xiao, Zhongqian Fu, Fenglong Song** in CVPR 2021.

It is a **Pytorch** implementation.

# Paper
- https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2021/supplemental/Maggioni_Efficient_Multi-Stage_Video_CVPR_2021_supplemental.pdf

# Requirements
1. PyTorch>=1.6
2. Numpy
3. scikti-image
4. tensorboardX (for visualization of loss, PSNR and SSIM)
5. torchstat （for computing GFLOPs）

# Code
1. `config.py` is the code for setting hyperparameters.
2. `dataset.py` and load_data.py is the code for loading data from dataset.
3. `train.py` is the code for training process
4. `inference.py` is the code for validation process.
5. `models.py` and `./isp/ISP_CNN.pth` is called by `inference.py` for converting .tiff to .png, which refer to the code RViDeNet(https://github.com/cao-cong/RViDeNet).

# Dataset
CRVD Dataset (https://github.com/cao-cong/RViDeNet)

# Usage
modify `data_root` in `config.py`, and `gt_name/noisy_name` in function `decode_data` in`load_data.py`, and run `train.py` for training process. After convergence， run `inference.py` for validation process.

# Results
ISO average raw psnr:42.02, iso frame average raw ssim:0.9800 in CRVD datasets (~5.38GFLPs), which is still lower than the experiment results mentioned in paper. 

# Acknowledgement
This implementations are inspired by following projects:
- [RViDeNet]  (https://github.com/cao-cong/RViDeNet)
 
 
*Many thanks for coming here! It will be highly appreciated if you offer any suggestion.

Support me by starring or forking this repo., please.*
