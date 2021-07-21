import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg

device = cfg.device
# device = 'cpu'

cfa = np.array(
    [[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.65, 0.2784, -0.2784, -0.65], [-0.2784, 0.65, -0.65, 0.2764]])

cfa = np.expand_dims(cfa, axis=2)
cfa = np.expand_dims(cfa, axis=3)
cfa = torch.tensor(cfa).float()  # .cuda()
cfa_inv = cfa.transpose(0, 1)

# dwt dec
h0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
h1 = np.array([-1 / math.sqrt(2), 1 / math.sqrt(2)])
h0 = np.array(h0[::-1]).ravel()
h1 = np.array(h1[::-1]).ravel()
h0 = torch.tensor(h0).float().reshape((1, 1, -1))
h1 = torch.tensor(h1).float().reshape((1, 1, -1))
h0_col = h0.reshape((1, 1, -1, 1))  # col lowpass
h1_col = h1.reshape((1, 1, -1, 1))  # col highpass
h0_row = h0.reshape((1, 1, 1, -1))  # row lowpass
h1_row = h1.reshape((1, 1, 1, -1))  # row highpass
ll_filt = torch.cat([h0_row, h1_row], dim=0)

# dwt rec
g0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
g1 = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
g0 = np.array(g0).ravel()
g1 = np.array(g1).ravel()
g0 = torch.tensor(g0).float().reshape((1, 1, -1))
g1 = torch.tensor(g1).float().reshape((1, 1, -1))
g0_col = g0.reshape((1, 1, -1, 1))
g1_col = g1.reshape((1, 1, -1, 1))
g0_row = g0.reshape((1, 1, 1, -1))
g1_row = g1.reshape((1, 1, 1, -1))


class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa)

    def forward(self, x):
        out = self.net1(x)
        return out


class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa_inv)

    def forward(self, x):
        out = self.net1(x)
        return out


class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        self.net1 = nn.Conv2d(1, 2, kernel_size=(1, 2), stride=(1, 2), padding=0,
                              bias=None)  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = torch.nn.Parameter(ll_filt)  # torch.Size([2, 1, 1, 2])

    def forward(self, x):
        B, C, H, W = x.shape
        ll = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        hl = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        lh = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        hh = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)

        for i in range(C):
            ll_ = self.net1(x[:, i:(i + 1) * 1, :, :])  # 1 * 2 * 128 * 64
            y = []
            for j in range(2):
                weight = self.net1.weight.transpose(2, 3)
                y_out = F.conv2d(ll_[:, j:(j + 1) * 1, :, :], weight, stride=(2, 1), padding=0, bias=None)
                y.append(y_out)
            y_ = torch.cat([y[0], y[1]], dim=1)
            ll[:, i:(i + 1), :, :] = y_[:, 0:1, :, :]
            hl[:, i:(i + 1), :, :] = y_[:, 1:2, :, :]
            lh[:, i:(i + 1), :, :] = y_[:, 2:3, :, :]
            hh[:, i:(i + 1), :, :] = y_[:, 3:4, :, :]

        out = torch.cat([ll, hl, lh, hh], dim=1)
        return out


class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(g0_col)  # torch.Size([1,1,2,1])
        self.net2 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net2.weight = torch.nn.Parameter(g1_col)  # torch.Size([1,1,2,1])

    def forward(self, x):
        lls = x[:, 0:4, :, :]
        hls = x[:, 4:8, :, :]
        lhs = x[:, 8:12, :, :]
        hhs = x[:, 12:16, :, :]
        B, C, H, W = lls.shape
        out = torch.ones([B, C, int(H * 2), int(W * 2)], device=device)
        for i in range(C):
            ll = lls[:, i:i + 1, :, :]
            hl = hls[:, i:i + 1, :, :]
            lh = lhs[:, i:i + 1, :, :]
            hh = hhs[:, i:i + 1, :, :]

            lo = self.net1(ll) + self.net2(hl)  # 1 * 1 * 128 * 64
            hi = self.net1(lh) + self.net2(hh)  # 1 * 1 * 128 * 64
            weight_l = self.net1.weight.transpose(2, 3)
            weight_h = self.net2.weight.transpose(2, 3)
            l = F.conv_transpose2d(lo, weight_l, stride=(1, 2), padding=0, bias=None)
            h = F.conv_transpose2d(hi, weight_h, stride=(1, 2), padding=0, bias=None)
            out[:, i:i + 1, :, :] = l + h
        return out


class Fusion_down(nn.Module):
    def __init__(self):
        super(Fusion_down, self).__init__()
        self.net1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Fusion_up(nn.Module):
    def __init__(self):
        super(Fusion_up, self).__init__()
        self.net1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Denoise_down(nn.Module):

    def __init__(self):
        super(Denoise_down, self).__init__()
        self.net1 = nn.Conv2d(21, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out


class Denoise_up(nn.Module):

    def __init__(self):
        super(Denoise_up, self).__init__()
        self.net1 = nn.Conv2d(25, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out


class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(33, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class VideoDenoise(nn.Module):
    def __init__(self):
        super(VideoDenoise, self).__init__()

        self.fusion = Fusion_down()
        self.denoise = Denoise_down()

    def forward(self, ft0, ft1, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = torch.clamp(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - gamma)) + torch.mul(ft1, gamma)

        # denoise
        sigma_ll0 = torch.clamp(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)
        return gamma, denoise_out


class MultiVideoDenoise(nn.Module):
    def __init__(self):
        super(MultiVideoDenoise, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up()

    def forward(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = torch.clamp(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - gamma)) + torch.mul(ft1, gamma)

        # denoise
        sigma_ll0 = torch.clamp(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, denoise_down, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        return gamma, fusion_out, denoise_out, sigma


class MainDenoise(nn.Module):
    def __init__(self):
        super(MainDenoise, self).__init__()
        self.ct = ColorTransfer()
        self.cti = ColorTransferInv()
        self.ft = FreTransfer()
        self.fti = FreTransferInv()
        self.vd = VideoDenoise()
        self.md1 = MultiVideoDenoise()
        self.md0 = MultiVideoDenoise()
        self.refine = Refine()

    def transform(self, x):
        net1 = self.ct(x)
        out = self.ft(net1)
        return out

    def transforminv(self, x):
        net1 = self.fti(x)
        out = self.cti(net1)
        return out

    def forward(self, x, coeff_a=1, coeff_b=1):
        ft0 = x[:, 0:4, :, :]  # 1*4*128*128, the t-1 fusion frame
        ft1 = x[:, 4:8, :, :]  # 1*4*128*128, the t frame

        ft0_d0 = self.transform(ft0)        # scale0, torch.Size([1, 16, 256, 256])
        ft1_d0 = self.transform(ft1)

        ft0_d1 = self.ft(ft0_d0[:,0:4,:,:])     # scale1,torch.Size([1, 16, 128, 128])
        ft1_d1 = self.ft(ft1_d0[:, 0:4, :, :])

        ft0_d2 = self.ft(ft0_d1[:,0:4,:,:])     # scale2, torch.Size([1, 16, 64, 64])
        ft1_d2 = self.ft(ft1_d1[:, 0:4, :, :])


        gamma, denoise_out = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)
        denoise_out_d2 = self.fti(denoise_out)
        gamma_up_d2 = F.upsample(gamma, scale_factor=2)


        gamma, fusion_out, denoise_out, sigma = self.md1(ft0_d1, ft1_d1, gamma_up_d2, denoise_out_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti(denoise_out)
        gamma_up_d1 = F.upsample(gamma, scale_factor=2)

        gamma, fusion_out, denoise_out, sigma = self.md0(ft0_d0, ft1_d0, gamma_up_d1, denoise_up_d1, coeff_a, coeff_b)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out, sigma], axis=1)  # 1 * 36 * 128 * 128
        omega = self.refine(refine_in)  # 1 * 16 * 128 * 128
        refine_out = torch.mul(denoise_out, (1 - omega)) + torch.mul(fusion_out, omega)

        fusion_out = self.transforminv(fusion_out)
        refine_out = self.transforminv(refine_out)
        denoise_out = self.transforminv(denoise_out)

        return gamma, fusion_out, denoise_out, omega, refine_out



