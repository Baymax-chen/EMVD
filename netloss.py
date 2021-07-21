import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, predict, label):
        l1loss = torch.mean(torch.abs(predict - label))
        return l1loss

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, image, label):
        MSE = (image - label) * (image - label)
        MSE = torch.mean(MSE)
        PSNR = 10 * torch.log(1 / MSE) / torch.log(torch.Tensor([10.])).cuda()  # torch.log is log base e

        return PSNR


def loss_color(model, layers, device):       # Color Transform
    '''
    :param model:
    :param layers: layer name we want to use orthogonal regularization
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    ct = params['ct.net1.weight'].squeeze()
    cti = params['cti.net1.weight'].squeeze()
    weight_squared = torch.matmul(ct, cti)
    diag = torch.eye(weight_squared.shape[0], dtype=torch.float32, device=device)
    loss = ((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def loss_wavelet(model, device):                            # Frequency Transform
    '''
    :param model:
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    ft = params['ft.net1.weight'].squeeze()
    fti = torch.cat([params['fti.net1.weight'],params['fti.net2.weight']],dim= 0).squeeze()
    weight_squared = torch.matmul(ft, fti)
    diag = torch.eye(weight_squared.shape[1], dtype=torch.float32, device=device)
    loss=((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth
