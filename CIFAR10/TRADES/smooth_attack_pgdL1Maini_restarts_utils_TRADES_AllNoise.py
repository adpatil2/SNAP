from __future__ import print_function
import os
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import numpy as np
from typing import Union
import random


def eval_adv_test_whitebox(model, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    batch_ind=0

    print("selected epsilon")
    print(args.epsilon)

    print("selected no. of steps")
    print(args.num_steps)

    print("attack_type")
    print(args.attack_type)

    device = args.device

    if args.attack_type == 'l1':

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust, eta_final_batch, err_nat_vec, err_rob_vec = pgd_l1_topk(model, X, y, args) # , eta_final_batch
            robust_err_total += err_robust
            natural_err_total += err_natural
            batch_ind += 1
            print("batch_ind:{}".format(batch_ind))

            eta_final_batch_flat = eta_final_batch.view(eta_final_batch.size(0),-1)

            if 'stored_pert_vecs' in locals():
                stored_pert_vecs = torch.cat((stored_pert_vecs,eta_final_batch_flat),dim=0)
                stored_err_nat_vec = torch.cat((stored_err_nat_vec,err_nat_vec),dim=0)
                stored_err_rob_vec = torch.cat((stored_err_rob_vec,err_rob_vec),dim=0)
            else: 
                stored_pert_vecs = eta_final_batch_flat
                stored_err_nat_vec = err_nat_vec
                stored_err_rob_vec = err_rob_vec

            print("Size of stored_pert_vec:")
            print(stored_pert_vecs.size(0))
            print(stored_pert_vecs.size(1))

            if batch_ind*args.test_batch_size >= args.num_tests: 
                print("Finished {} number of tests".format(args.num_tests))
                break

    elif args.attack_type == 'l2':

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust, eta_final_batch, err_nat_vec, err_rob_vec = _pgd_l2_whitebox(model, X, y, args) # , eta_final_batch
            robust_err_total += err_robust
            natural_err_total += err_natural
            batch_ind += 1
            print("batch_ind:{}".format(batch_ind))

            eta_final_batch_flat = eta_final_batch.view(eta_final_batch.size(0),-1)

            if 'stored_pert_vecs' in locals():
                stored_pert_vecs = torch.cat((stored_pert_vecs,eta_final_batch_flat),dim=0)
                stored_err_nat_vec = torch.cat((stored_err_nat_vec,err_nat_vec),dim=0)
                stored_err_rob_vec = torch.cat((stored_err_rob_vec,err_rob_vec),dim=0)
            else: 
                stored_pert_vecs = eta_final_batch_flat
                stored_err_nat_vec = err_nat_vec
                stored_err_rob_vec = err_rob_vec

            print("Size of stored_pert_vec:")
            print(stored_pert_vecs.size(0))
            print(stored_pert_vecs.size(1))

            if batch_ind*args.test_batch_size >= args.num_tests: 
                print("Finished {} number of tests".format(args.num_tests))
                break

    elif args.attack_type == 'linf':

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust, eta_final_batch, err_nat_vec, err_rob_vec = _pgd_linf_whitebox(model, X, y, args) # , eta_final_batch
            robust_err_total += err_robust
            natural_err_total += err_natural
            batch_ind += 1
            print("batch_ind:{}".format(batch_ind))

            eta_final_batch_flat = eta_final_batch.view(eta_final_batch.size(0),-1)

            if 'stored_pert_vecs' in locals():
                stored_pert_vecs = torch.cat((stored_pert_vecs,eta_final_batch_flat),dim=0)
                stored_err_nat_vec = torch.cat((stored_err_nat_vec,err_nat_vec),dim=0)
                stored_err_rob_vec = torch.cat((stored_err_rob_vec,err_rob_vec),dim=0)
            else: 
                stored_pert_vecs = eta_final_batch_flat
                stored_err_nat_vec = err_nat_vec
                stored_err_rob_vec = err_rob_vec

            print("Size of stored_pert_vec:")
            print(stored_pert_vecs.size(0))
            print(stored_pert_vecs.size(1))

            if batch_ind*args.test_batch_size >= args.num_tests: 
                print("Finished {} number of tests".format(args.num_tests))
                break

    natural_acc_fraction = 1 - natural_err_total.item()/float(batch_ind*args.test_batch_size)
    robust_acc_fraction = 1 - robust_err_total.item()/float(batch_ind*args.test_batch_size)
    print('natural_acc_fraction (test): ', natural_acc_fraction)
    print('robust_acc_fraction (test): ', robust_acc_fraction)

    return natural_acc_fraction, robust_acc_fraction, stored_pert_vecs.cpu(), stored_err_nat_vec.cpu(), stored_err_rob_vec.cpu()


def pgd_l1_topk(model, X,y, args, alpha = 0.05, k = 20, restarts = 1, version = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    
    epsilon = args.epsilon
    smth_avg_steps = args.smth_avg_steps
    num_avg_steps = args.grad_avg_steps
    num_iter = args.num_steps
    restarts = args.num_restarts

    device = args.device


    gap = alpha
    max_delta = torch.zeros_like(X)
    
    for r in range(restarts):

        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)

    
        for t in range (num_iter):
            for step in range(smth_avg_steps):
                out = model(X.data+delta.data, args)
                if step != 0:
                    cum_counts = cum_counts + (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                else:
                    cum_counts = (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()

            incorrect = cum_counts.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            for avg_ in range(num_avg_steps):
                noi_z = model(X+delta, args)
                soft_z = F.softmax(noi_z, dim=1)
                if avg_ == 0:
                    soft_z_avg = soft_z
                else:
                    soft_z_avg = soft_z_avg + soft_z

            soft_z_avg = soft_z_avg/float(num_avg_steps) 
            logsoftmax = torch.log(soft_z_avg.clamp(min=1e-20))
            loss = F.nll_loss(logsoftmax, y)
            loss.backward()
            k = random.randint(5,20)
            alpha = 0.05/k*20
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 

        with torch.no_grad():
            for step in range(smth_avg_steps):

                out_pgd = model(X.data+delta.data, args)

                if step != 0:
                    cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
                else:
                    cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()

            incorrect = cum_counts_pgd.data.max(1)[1] != y.data

        #Edit Max Delta only for successful attacks ... only after zeroth iteration.... 
        if r != 0:
            max_delta[incorrect] = delta.detach()[incorrect] 
        else:
            max_delta = delta.detach()

    with torch.no_grad():
        for step in range(smth_avg_steps):

            out = model(X.data, args)
            out_pgd = model(X.data+max_delta.data, args)

            if step != 0:
                cum_counts = cum_counts + (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
            else:
                cum_counts = (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()

        err = (  cum_counts.data.max(1)[1] != y.data).float().sum()
        err_pgd = (cum_counts_pgd.data.max(1)[1] != y.data).float().sum()
        err_vec = (  cum_counts.data.max(1)[1] != y.data).float()
        err_pgd_vec = (cum_counts_pgd.data.max(1)[1] != y.data).float()
        #eta_final = X_pgd.data - X.data
        print('err nat: ', err)
        print('err pgd (white-box): ', err_pgd)

    return  err, err_pgd, max_delta, err_vec, err_pgd_vec


def _pgd_l2_whitebox(model,X,y, args):

    epsilon = args.epsilon
    num_steps = args.num_steps
    step_size = epsilon * args.step_size_fact # epsilon*8/num_steps
    smth_avg_steps = args.smth_avg_steps
    num_avg_steps = args.grad_avg_steps
    restarts = args.num_restarts

    device = args.device

    print("epsilon is:{}".format(epsilon))
    print("step_size is:{}".format(step_size))

    
    for k in range(restarts):
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
        
        delta = X_pgd.data - X.data
        with torch.no_grad():
            for step in range(smth_avg_steps):
                out_pgd = model(X.data+delta.data, args)
                if step != 0:
                    cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
                else:
                    cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()

            incorrect = cum_counts_pgd.data.max(1)[1] != y.data

        #Edit Max Delta only for successful attacks ... only after zeroth iteration.... 
        if k != 0:
            max_delta[incorrect] = delta.detach()[incorrect] 
        else:
            max_delta = delta.detach()
    
    with torch.no_grad():
        for step in range(smth_avg_steps):

            out = model(X.data, args)
            out_pgd = model(X.data+max_delta.data, args)

            if step != 0:
                cum_counts = cum_counts + (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
            else:
                cum_counts = (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()


        err = (  cum_counts.data.max(1)[1] != y.data).float().sum()
        err_pgd = (cum_counts_pgd.data.max(1)[1] != y.data).float().sum()
        err_vec = (  cum_counts.data.max(1)[1] != y.data).float()
        err_pgd_vec = (cum_counts_pgd.data.max(1)[1] != y.data).float()
        #eta_final = max_delta
        print('err nat: ', err)
        print('err pgd (white-box): ', err_pgd)

    return err, err_pgd, max_delta, err_vec, err_pgd_vec


def _pgd_linf_whitebox(model,X,y, args):

    epsilon = args.epsilon
    num_steps = args.num_steps
    step_size = epsilon * args.step_size_fact #step_size = epsilon*8/num_steps
    smth_avg_steps = args.smth_avg_steps
    num_avg_steps = args.grad_avg_steps
    restarts = args.num_restarts

    device = args.device

    print("epsilon is:{}".format(epsilon))
    print("step_size is:{}".format(step_size))


    for k in range(restarts):
        X_pgd = Variable(X.data, requires_grad=True)
        if args.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

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
            eta = step_size * X_pgd_grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        delta = X_pgd.data - X.data
        with torch.no_grad():
            for step in range(smth_avg_steps):
                out_pgd = model(X.data+delta.data, args)
                if step != 0:
                    cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
                else:
                    cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()

            incorrect = cum_counts_pgd.data.max(1)[1] != y.data

        #Edit Max Delta only for successful attacks ... only after zeroth iteration.... 
        if k != 0:
            max_delta[incorrect] = delta.detach()[incorrect] 
        else:
            max_delta = delta.detach()

    with torch.no_grad():
        for step in range(smth_avg_steps):

            out = model(X.data, args)
            out_pgd = model(X.data+max_delta.data, args)

            if step != 0:
                cum_counts = cum_counts + (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = cum_counts_pgd + (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()
            else:
                cum_counts = (torch.max(out.data,dim=1,keepdim=True)[0].repeat(1,out.data.size(1)) == out.data).float()
                cum_counts_pgd = (torch.max(out_pgd.data,dim=1,keepdim=True)[0].repeat(1,out_pgd.data.size(1)) == out_pgd.data).float()


        err = (  cum_counts.data.max(1)[1] != y.data).float().sum()
        err_pgd = (cum_counts_pgd.data.max(1)[1] != y.data).float().sum()
        err_vec = (  cum_counts.data.max(1)[1] != y.data).float()
        err_pgd_vec = (cum_counts_pgd.data.max(1)[1] != y.data).float()
        #eta_final = max_delta
        print('err nat: ', err)
        print('err pgd (white-box): ', err_pgd)

    return err, err_pgd, max_delta, err_vec, err_pgd_vec

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

def percentile(t: torch.tensor, q: float): # -> Union[int, float]
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
