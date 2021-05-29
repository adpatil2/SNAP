from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
import time
import random

import numpy as np
import numpy.matlib as np_mat
from Utils_Train_Logistics import AverageMeter, accuracy





def eval_train(model, train_loader, args):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data,args)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def eval_test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data, args)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def _record_eta_batchwise(model,X,y, args):

    epsilon = args.epsilon_attack
    num_steps = args.num_steps_attack
    step_size = epsilon*8/num_steps
    smth_avg_steps = args.smth_avg_steps
    num_avg_steps = args.grad_avg_steps

    device = args.device

    print("epsilon is:{}".format(epsilon))
    print("num_steps is:{}".format(num_steps))

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).normal_(mean=0,std=2*epsilon).to(device) #.uniform_(-epsilon, epsilon).to(device)
        random_noise_reshaped = random_noise.view(random_noise.size(0),-1)
        random_noise_reshaped_norm = torch.norm(random_noise_reshaped,p=2,dim=1,keepdim=True)
        all_epsilon_vec = (epsilon*torch.ones([random_noise_reshaped_norm.size(0),random_noise_reshaped_norm.size(1)])).type_as(random_noise_reshaped_norm)
        random_noise_reshaped_normzed = epsilon*torch.div(random_noise_reshaped, torch.max(random_noise_reshaped_norm,all_epsilon_vec).expand(-1,random_noise_reshaped.size(1)) +1e-8)
        random_noise_final = random_noise_reshaped_normzed.view(X_pgd.size(0),X_pgd.size(1),X_pgd.size(2),X_pgd.size(3))

        X_pgd = Variable(X_pgd.data + random_noise_final, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        ### Here you add averaging... 
        for avg_ in range(num_avg_steps):
            noi_z = model(X_pgd, args)
            soft_z = F.softmax(noi_z, dim=1)
            if avg_ == 0:
                soft_z_avg = soft_z
            else:
                soft_z_avg = soft_z_avg + soft_z

        soft_z_avg = soft_z_avg/float(num_avg_steps) 
        logsoftmax = torch.log(soft_z_avg.clamp(min=1e-20))
        loss = F.nll_loss(logsoftmax, y)
        loss.backward()
        X_pgd_grad = X_pgd.grad.data
        X_pgd_grad_reshaped = X_pgd_grad.view(X_pgd_grad.size(0),-1)
        X_pgd_grad_reshaped_norm = torch.norm(X_pgd_grad_reshaped,p=2,dim=1,keepdim=True)
        X_pgd_grad_reshaped_normzed = torch.div(X_pgd_grad_reshaped, X_pgd_grad_reshaped_norm.expand(-1,X_pgd_grad_reshaped.size(1)) +1e-8)
        X_pgd_grad_normzed = X_pgd_grad_reshaped_normzed.view(X_pgd_grad.size(0),X_pgd_grad.size(1),X_pgd_grad.size(2),X_pgd_grad.size(3))
        eta = step_size * X_pgd_grad_normzed.data

        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        eta_tot = X_pgd.data - X.data

        eta_tot_reshaped = eta_tot.view(eta_tot.size(0),-1)
        eta_tot_reshaped_norm = torch.norm(eta_tot_reshaped,p=2,dim=1,keepdim=True)
        all_epsilon_vec = (epsilon*torch.ones([eta_tot_reshaped_norm.size(0),eta_tot_reshaped_norm.size(1)])).type_as(eta_tot_reshaped_norm)
        eta_tot_reshaped_normzed = epsilon*torch.div(eta_tot_reshaped, torch.max(eta_tot_reshaped_norm,all_epsilon_vec).expand(-1,eta_tot_reshaped.size(1)) +1e-8)
        eta_tot_final = eta_tot_reshaped_normzed.view(X_pgd_grad.size(0),X_pgd_grad.size(1),X_pgd_grad.size(2),X_pgd_grad.size(3))

        X_pgd = Variable(torch.clamp(X.data + eta_tot_final.data, 0, 1.0), requires_grad=True)        
        #X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    
    
    with torch.no_grad():
        for step in range(smth_avg_steps):

            out = model(X.data, args)
            out_pgd = model(X_pgd.data, args)

            if step != 0:
                cum_counts = cum_counts + (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
            else:
                cum_counts = (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()


        err = (  cum_counts.data.max(1)[1] != y.data).float().sum()
        err_pgd = (cum_counts_pgd.data.max(1)[1] != y.data).float().sum()
        eta_final = X_pgd.data - X.data
        print('err nat: ', err)
        print('err pgd (white-box): ', err_pgd)

    return X_pgd.data, err_pgd, eta_final

def eval_adv_train_whitebox(model, epoch_no, train_loader,args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    rob_err_train_tot = 0
    nat_err_train_tot = 0
    batch_count = 0

    for data, target in train_loader:
        batch_size = len(data)

        data, target = data.to(args.device), target.to(args.device)
        #pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        adv_x, rob_err_train, eta_final_batch = _record_eta_batchwise(model, X, y, args)
        rob_err_train_tot += rob_err_train

        ## Okay, here we need to check the basis for projections.... 

        if args.noise_shp_basis == 'std':

            eta_mean_sq_proj_batch = torch.mean(torch.mul(eta_final_batch,eta_final_batch),dim=0)

        elif args.noise_shp_basis == 'image':

            eta_final_batch_flat = eta_final_batch.view(eta_final_batch.size(0),-1)
            eta_proj_batch_ele = torch.mm(eta_final_batch_flat,args.vecs_SS_noisy_pt)
            eta_sq_proj_batch_ele = torch.mul(eta_proj_batch_ele,eta_proj_batch_ele)
            eta_mean_sq_proj_batch = torch.mean(eta_sq_proj_batch_ele,dim=0)

        if 'stored_eta_mean_sq_proj_final' in locals(): 
            stored_eta_mean_sq_proj_final = stored_eta_mean_sq_proj_final + eta_mean_sq_proj_batch.data
        else:
            stored_eta_mean_sq_proj_final = eta_mean_sq_proj_batch.data

        print(batch_count)

        ### Here you can potentially break the loop... 
        ## Run it only as a partial epoch.. 
        if batch_count == args.ssn_epoch_batches: 
            break
        batch_count = batch_count+1

    rob_train_acc_frac = 1 - rob_err_train_tot.item()/float((batch_count+1)*eta_final_batch.size(0))
    print('robust_acc_fraction (train): ', rob_train_acc_frac)
    return stored_eta_mean_sq_proj_final, rob_train_acc_frac

## this is modified according to TRADES.....
def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 20) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).half() * grad_check.sign().half() #.half()  .half() .half()).half()
    return k_hot.view(batch_size, channels, pix, pix).float()


def proj_l1ball(x, epsilon=10, device = "cuda:0"):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y


def proj_simplex(v, s=1, device = "cuda:0"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).float().to(device) #.half()
    comp = (vec > (cssv - s)).half()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.half() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta.float()).clamp(min=0)
    return w