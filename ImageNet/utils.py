from __future__ import print_function
import os
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import yaml
import math
# from easydict import EasyDict
import shutil


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))
        # self.means = torch.tensor(means) #.cuda()
        # self.sds = torch.tensor(sds)# .cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

def get_normalize_layer() -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    return NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class NoiseGenAndProjLayer(nn.Module):
    """
    Main Class
    """
    def __init__(self, vecs_SS_noisy_pt,DimWise_noi_std_pt_ch0,DimWise_noi_std_pt_ch1,DimWise_noi_std_pt_ch2):
        """
        Constructor
        """
        super().__init__()
        self.register_buffer('vecs_SS_noisy_pt', vecs_SS_noisy_pt)
        self.register_buffer('DimWise_noi_std_pt_ch0', DimWise_noi_std_pt_ch0)
        self.register_buffer('DimWise_noi_std_pt_ch1', DimWise_noi_std_pt_ch1)
        self.register_buffer('DimWise_noi_std_pt_ch2', DimWise_noi_std_pt_ch2)

    def forward(self, x):

        if len(list(x.size())) == 4:

            noi_std_tensor_mat_ch0 = self.DimWise_noi_std_pt_ch0.expand(-1,x.size(0))
            noi_std_tensor_mat_ch1 = self.DimWise_noi_std_pt_ch1.expand(-1,x.size(0))
            noi_std_tensor_mat_ch2 = self.DimWise_noi_std_pt_ch2.expand(-1,x.size(0))
            noi_std_tensor_mat_all = torch.cat((noi_std_tensor_mat_ch0,noi_std_tensor_mat_ch1,noi_std_tensor_mat_ch2),dim=1)

            # print("no of inputs here is: {}".format(x.size(0)))

            proj_noise_reshaped_tr_all = torch.mul(noi_std_tensor_mat_all,torch.randn_like(noi_std_tensor_mat_all))

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            noise_reshaped_tr_all = torch.mm(self.vecs_SS_noisy_pt, proj_noise_reshaped_tr_all  )
            # end.record()
            # torch.cuda.synchronize()
            # print("Noise Svecs MVM time: {}".format(start.elapsed_time(end)))

            noise_reshaped_ch0 = noise_reshaped_tr_all[:,:x.size(0)].transpose(1,0)
            noise_reshaped_ch1 = noise_reshaped_tr_all[:,x.size(0):2*x.size(0)].transpose(1,0)
            noise_reshaped_ch2 = noise_reshaped_tr_all[:,2*x.size(0):].transpose(1,0)
            noise_ch0 = noise_reshaped_ch0.view(x.size(0),x.size(2),x.size(3))
            noise_ch1 = noise_reshaped_ch1.view(x.size(0),x.size(2),x.size(3))
            noise_ch2 = noise_reshaped_ch2.view(x.size(0),x.size(2),x.size(3))
            noise = torch.cat(( torch.unsqueeze(noise_ch0,dim=1), torch.unsqueeze(noise_ch1,dim=1), torch.unsqueeze(noise_ch2,dim=1) ), dim=1)

            return x + noise.float()

        elif len(list(x.size())) == 2:

            ## Here x is eta_final_batch_all_flat
            ## Previously we were doing this in float() precision... now we have shifted this to half() precision....
            eta_sq_proj_batch_all_ele = torch.mm(x.half(),self.vecs_SS_noisy_pt)**2

            return eta_sq_proj_batch_all_ele.float()



class NoiseLayer(nn.Module):
    """
    Main Class
    """

    def __init__(self, args):
        """
        Constructor
        """
        super().__init__()
        self.register_buffer('DimWise_noi_std_pt', args.DimWise_noi_std_pt)
        self.register_buffer('unit_std_scale', args.unit_std_scale)
        self.model_m_dist = args.m_dist


    def forward(self, x):

        unit_var_noise = torch.mul(self.unit_std_scale , torch.squeeze( self.model_m_dist.sample(x.data.size()) ) )  
        final_recon_x = x + torch.unsqueeze(self.DimWise_noi_std_pt,0) * unit_var_noise ## We need to send the noise to GPU... 

        return final_recon_x


def save_checkpoint(state, is_best, filepath, epoch):
    filename = os.path.join(filepath, 'resnet_checkpoint_epoch{}.pth.tar'.format(epoch)) # os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'resnet_best.pth.tar'))

def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)

