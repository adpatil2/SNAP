import os
import shutil

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time
import sys
import datetime
from utils import *


def test(loader: DataLoader, model: torch.nn.Module, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    noise_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            
            inputs = inputs.cuda()
            targets = targets.cuda()

            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)


## This is your function for recording \eta vectors.... 
def _record_eta_batchwise(model, X, y, args):

    epsilon=args.epsilon_attack
    num_steps=args.num_steps_attack
    step_size = epsilon*0.8

    X_pgd = Variable(X.data, requires_grad=True)
    model.eval()
    if args.random_start:
        with torch.no_grad():
            random_noise = torch.FloatTensor(*X_pgd.shape).normal_(mean=0,std=2*epsilon).detach().cuda() #.uniform_(-epsilon, epsilon).to(device)
            random_noise_reshaped = random_noise.view(random_noise.size(0),-1)
            random_noise_reshaped_norm = torch.norm(random_noise_reshaped,p=2,dim=1,keepdim=True)
            all_epsilon_vec = (epsilon*torch.ones([random_noise_reshaped_norm.size(0),random_noise_reshaped_norm.size(1)])).type_as(random_noise_reshaped_norm)
            random_noise_reshaped_normzed = epsilon*torch.div(random_noise_reshaped, torch.max(random_noise_reshaped_norm,all_epsilon_vec).expand(-1,random_noise_reshaped.size(1)) +1e-8)
            random_noise_final = random_noise_reshaped_normzed.view(X_pgd.size(0),X_pgd.size(1),X_pgd.size(2),X_pgd.size(3))

        X_pgd = Variable(X_pgd.data + random_noise_final.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            #loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        X_pgd_grad = X_pgd.grad.data
        with torch.no_grad():
            X_pgd_grad_reshaped = X_pgd_grad.view(X_pgd_grad.size(0),-1)
            X_pgd_grad_reshaped_norm = torch.norm(X_pgd_grad_reshaped,p=2,dim=1,keepdim=True)
            X_pgd_grad_reshaped_normzed = torch.div(X_pgd_grad_reshaped, X_pgd_grad_reshaped_norm.expand(-1,X_pgd_grad_reshaped.size(1)) +1e-8)
            X_pgd_grad_normzed = X_pgd_grad_reshaped_normzed.view(X_pgd_grad.size(0),X_pgd_grad.size(1),X_pgd_grad.size(2),X_pgd_grad.size(3))
            eta = step_size * X_pgd_grad_normzed.data

            X_pgd = X_pgd.data + eta #, requires_grad=True)  Variable(

            eta_tot = X_pgd.data - X.data

            eta_tot_reshaped = eta_tot.view(eta_tot.size(0),-1)
            eta_tot_reshaped_norm = torch.norm(eta_tot_reshaped,p=2,dim=1,keepdim=True)
            all_epsilon_vec = (epsilon*torch.ones([eta_tot_reshaped_norm.size(0),eta_tot_reshaped_norm.size(1)])).type_as(eta_tot_reshaped_norm)
            eta_tot_reshaped_normzed = epsilon*torch.div(eta_tot_reshaped, torch.max(eta_tot_reshaped_norm,all_epsilon_vec).expand(-1,eta_tot_reshaped.size(1)) +1e-8)
            eta_tot_final = eta_tot_reshaped_normzed.view(X_pgd_grad.size(0),X_pgd_grad.size(1),X_pgd_grad.size(2),X_pgd_grad.size(3))

        X_pgd = Variable(torch.clamp(X.data + eta_tot_final.data, 0, 1.0), requires_grad=True)        
        #X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    with torch.no_grad():
        eta_final = X_pgd.data - X.data

    ## You need to remove this .cpu() on this line.... 
    return eta_final #.cpu()

## this is your function for doing extra epoch to collect adv. perturbation statistics.. 
def eval_adv_train_whitebox(model, epoch_no, train_loader, args):

    eta_comp_time = AverageMeter()
    eta_proj_time = AverageMeter()
    eta_stor_time = AverageMeter()

    model.eval()
    rob_err_train_tot = 0
    nat_err_train_tot = 0
    batch_count = 0

    for data, target in train_loader:
        batch_size = len(data)

        data, target = data.cuda(), target.cuda()
        #pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        batch_start_time = time.time()
        eta_final_batch = _record_eta_batchwise(model, X, y, args)
        batch_end_time = time.time()
        print(batch_count)

        eta_mean_sq_proj_batch = torch.mean(torch.mul(eta_final_batch,eta_final_batch),dim=0)

        if 'stored_eta_mean_sq_proj_final' in locals(): 
            stored_eta_mean_sq_proj_final = stored_eta_mean_sq_proj_final + eta_mean_sq_proj_batch.data
        else:
            stored_eta_mean_sq_proj_final = eta_mean_sq_proj_batch.data

        
        assert not torch.isnan(eta_mean_sq_proj_batch).any()

        
        ### Here you can potentially break the loop... 
        ## Run it only as a partial epoch.. 
        if batch_count == args.ssn_epoch_batches: 
            break

        batch_count = batch_count+1

    return stored_eta_mean_sq_proj_final


