import numpy as np
import torch
from scipy.stats import poisson
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn.functional as F
import config as cfg
import torch.nn as nn

def pack_gbrg_raw(raw):
    #pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],          # r
                          im[1:H:2, 1:W:2, :],          # gr
                          im[0:H:2, 1:W:2, :],          # b
                          im[0:H:2, 0:W:2, :]), axis=2) # gb
    return out

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[0,i,j,3]        # gb
            output[2*i,2*j+1]=raw[0,i,j,2]      # b
            output[2*i+1,2*j]=raw[0,i,j,0]      # r
            output[2*i+1,2*j+1]=raw[0,i,j,1]    # gr
    return output


def compute_sigma(input, a, b):
    sigma = np.sqrt((input - 240) * a + b)
    return sigma


def preprocess(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def tensor2numpy(raw):  # raw: 1 * 4 * H * W
    input_full = raw.permute((0, 2, 3, 1))   # 1 * H * W * 4
    input_full = input_full.data.cpu().numpy()
    output = np.clip(input_full,0,1)
    return output

def pack_rggb_raw_for_compute_ssim(raw):

    im = raw.astype(np.float32)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def compute_ssim_for_packed_raw(raw1, raw2):
    raw1_pack = pack_rggb_raw_for_compute_ssim(raw1)
    raw2_pack = pack_rggb_raw_for_compute_ssim(raw2)
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

    return test_raw_ssim/4