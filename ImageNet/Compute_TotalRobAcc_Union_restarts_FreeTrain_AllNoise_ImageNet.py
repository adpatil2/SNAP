from __future__ import print_function
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import models
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np 
import math


parser = argparse.ArgumentParser(description='PyTorch IMAGENET inference code')

parser.add_argument('--noise-dist', type=str, default='laplace', help='Choices are laplace, uniform, gaussian, expsqrt')
parser.add_argument('--noise-pw', type=float, help='Choices are 60.0, 100.0, 150.0')
parser.add_argument('--num-steps', default=100, type=int, help='perturb number of steps')
parser.add_argument('--num-restarts', default=1, type=int, help='number of restarts')

args = parser.parse_args()

## Change for ErrVecs
vec_dir_dim = 'SmthPGD_ErrVecs/'
if not os.path.exists(vec_dir_dim):
    os.makedirs(vec_dir_dim)

res_dir_dim = 'SmoothPGD_Results/'
if not os.path.exists(res_dir_dim):
    os.makedirs(res_dir_dim)

if args.num_steps == 100:

    loaded_dict_l2 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l2Attack_Steps"+str(args.num_steps)+"Restarts"+str(args.num_restarts)+"Start1_ErrVecs.pt")
    stored_err_rob_vecs_l2_aft1 = loaded_dict_l2['stored_err_rob_vec']
    stored_err_nat_vecs_l2_aft1 = loaded_dict_l2['stored_err_nat_vec']
    eps_l2 = loaded_dict_l2['attack_eps']

    loaded_dict_l2 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l2Attack_Steps"+str(args.num_steps)+"Restarts"+str(args.num_restarts)+"Start251_ErrVecs.pt")
    stored_err_rob_vecs_l2_aft251 = loaded_dict_l2['stored_err_rob_vec']
    stored_err_nat_vecs_l2_aft251 = loaded_dict_l2['stored_err_nat_vec']
    eps_l2 = loaded_dict_l2['attack_eps']

    loaded_dict_l2 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l2Attack_Steps"+str(args.num_steps)+"Restarts"+str(args.num_restarts)+"Start501_ErrVecs.pt")
    stored_err_rob_vecs_l2_aft501 = loaded_dict_l2['stored_err_rob_vec']
    stored_err_nat_vecs_l2_aft501 = loaded_dict_l2['stored_err_nat_vec']
    eps_l2 = loaded_dict_l2['attack_eps']
    
    loaded_dict_l2 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l2Attack_Steps"+str(args.num_steps)+"Restarts"+str(args.num_restarts)+"Start751_ErrVecs.pt")
    stored_err_rob_vecs_l2_aft751 = loaded_dict_l2['stored_err_rob_vec']
    stored_err_nat_vecs_l2_aft751 = loaded_dict_l2['stored_err_nat_vec']
    eps_l2 = loaded_dict_l2['attack_eps']
    
    
    stored_err_rob_vecs_l2_all = torch.cat((stored_err_rob_vecs_l2_aft1[:250*50],stored_err_rob_vecs_l2_aft251[:250*50],stored_err_rob_vecs_l2_aft501[:250*50],stored_err_rob_vecs_l2_aft751[:250*50]))
    stored_err_nat_vecs_l2_all = torch.cat((stored_err_nat_vecs_l2_aft1[:250*50],stored_err_nat_vecs_l2_aft251[:250*50],stored_err_nat_vecs_l2_aft501[:250*50],stored_err_nat_vecs_l2_aft751[:250*50]))
    
    print("final length of stored_err_rob_vecs_l2_all:{}".format(stored_err_rob_vecs_l2_all.size(0)))
    print("final length of stored_err_nat_vecs_l2_all:{}".format(stored_err_nat_vecs_l2_all.size(0)))

    loaded_dict_linf = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_linfAttack_Steps"+str(args.num_steps)+"Epsilon0Restarts"+str(args.num_restarts)+"Start1_ErrVecs.pt")
    stored_err_rob_vecs_linf_aft1 = loaded_dict_linf['stored_err_rob_vec']
    stored_err_nat_vecs_linf_aft1 = loaded_dict_linf['stored_err_nat_vec']
    eps_linf = loaded_dict_linf['attack_eps']

    loaded_dict_linf = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_linfAttack_Steps"+str(args.num_steps)+"Epsilon0Restarts"+str(args.num_restarts)+"Start251_ErrVecs.pt")
    stored_err_rob_vecs_linf_aft251 = loaded_dict_linf['stored_err_rob_vec']
    stored_err_nat_vecs_linf_aft251 = loaded_dict_linf['stored_err_nat_vec']
    eps_linf = loaded_dict_linf['attack_eps']

    loaded_dict_linf = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_linfAttack_Steps"+str(args.num_steps)+"Epsilon0Restarts"+str(args.num_restarts)+"Start501_ErrVecs.pt")
    stored_err_rob_vecs_linf_aft501 = loaded_dict_linf['stored_err_rob_vec']
    stored_err_nat_vecs_linf_aft501 = loaded_dict_linf['stored_err_nat_vec']
    eps_linf = loaded_dict_linf['attack_eps']
    
    loaded_dict_linf = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_linfAttack_Steps"+str(args.num_steps)+"Epsilon0Restarts"+str(args.num_restarts)+"Start751_ErrVecs.pt")
    stored_err_rob_vecs_linf_aft751 = loaded_dict_linf['stored_err_rob_vec']
    stored_err_nat_vecs_linf_aft751 = loaded_dict_linf['stored_err_nat_vec']
    eps_linf = loaded_dict_linf['attack_eps']
    
    
    stored_err_rob_vecs_linf_all = torch.cat((stored_err_rob_vecs_linf_aft1[:250*50],stored_err_rob_vecs_linf_aft251[:250*50],stored_err_rob_vecs_linf_aft501[:250*50],stored_err_rob_vecs_linf_aft751[:250*50]))
    stored_err_nat_vecs_linf_all = torch.cat((stored_err_nat_vecs_linf_aft1[:250*50],stored_err_nat_vecs_linf_aft251[:250*50],stored_err_nat_vecs_linf_aft501[:250*50],stored_err_nat_vecs_linf_aft751[:250*50]))
    
    print("final length of stored_err_rob_vecs_linf_all:{}".format(stored_err_rob_vecs_linf_all.size(0)))
    print("final length of stored_err_nat_vecs_linf_all:{}".format(stored_err_nat_vecs_linf_all.size(0)))


    loaded_dict_l1 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l1Attack_Steps"+str(args.num_steps)+"Epsilon72Restarts"+str(args.num_restarts)+"Start1_ErrVecs.pt")
    stored_err_rob_vecs_l1_aft1 = loaded_dict_l1['stored_err_rob_vec']
    stored_err_nat_vecs_l1_aft1 = loaded_dict_l1['stored_err_nat_vec']
    eps_l1 = loaded_dict_l1['attack_eps']

    loaded_dict_l1 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l1Attack_Steps"+str(args.num_steps)+"Epsilon72Restarts"+str(args.num_restarts)+"Start251_ErrVecs.pt")
    stored_err_rob_vecs_l1_aft251 = loaded_dict_l1['stored_err_rob_vec']
    stored_err_nat_vecs_l1_aft251 = loaded_dict_l1['stored_err_nat_vec']
    eps_l1 = loaded_dict_l1['attack_eps']

    loaded_dict_l1 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l1Attack_Steps"+str(args.num_steps)+"Epsilon72Restarts"+str(args.num_restarts)+"Start501_ErrVecs.pt")
    stored_err_rob_vecs_l1_aft501 = loaded_dict_l1['stored_err_rob_vec']
    stored_err_nat_vecs_l1_aft501 = loaded_dict_l1['stored_err_nat_vec']
    eps_l1 = loaded_dict_l1['attack_eps']
    
    loaded_dict_l1 = torch.load(vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_l1Attack_Steps"+str(args.num_steps)+"Epsilon72Restarts"+str(args.num_restarts)+"Start751_ErrVecs.pt")
    stored_err_rob_vecs_l1_aft751 = loaded_dict_l1['stored_err_rob_vec']
    stored_err_nat_vecs_l1_aft751 = loaded_dict_l1['stored_err_nat_vec']
    eps_l1 = loaded_dict_l1['attack_eps']
    
    
    stored_err_rob_vecs_l1_all = torch.cat((stored_err_rob_vecs_l1_aft1[:250*50],stored_err_rob_vecs_l1_aft251[:250*50],stored_err_rob_vecs_l1_aft501[:250*50],stored_err_rob_vecs_l1_aft751[:250*50]))
    stored_err_nat_vecs_l1_all = torch.cat((stored_err_nat_vecs_l1_aft1[:250*50],stored_err_nat_vecs_l1_aft251[:250*50],stored_err_nat_vecs_l1_aft501[:250*50],stored_err_nat_vecs_l1_aft751[:250*50]))
    
    print("final length of stored_err_rob_vecs_l1_all:{}".format(stored_err_rob_vecs_l1_all.size(0)))
    print("final length of stored_err_nat_vecs_l1_all:{}".format(stored_err_nat_vecs_l1_all.size(0)))

    stored_err_rob_vecs_union = stored_err_rob_vecs_linf_all + stored_err_rob_vecs_l2_all + stored_err_rob_vecs_l1_all
    robust_union_acc = 1.0 - (stored_err_rob_vecs_union!=0).float().sum()/float(torch.numel(stored_err_rob_vecs_union))

    accuracy_list_file = open(res_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_Union_Robust_acc.txt","a")

    # print("natural accuracy:")
    # print(natural_avg_acc)
    print("robust union accuracy:")
    print(robust_union_acc)
    file_str_x = f"{eps_linf:5.5}\t{eps_l2:5.5}\t{eps_l1:5.5}\t{robust_union_acc:10.8}\n"

    accuracy_list_file.writelines(file_str_x)
    accuracy_list_file.close()
