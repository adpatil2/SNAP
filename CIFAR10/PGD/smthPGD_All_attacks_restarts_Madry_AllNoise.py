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

from smooth_attack_pgdL1Maini_restarts_utils_Madry_AllNoise import eval_adv_test_whitebox

from models.resnet import *   

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 inference code')

## Code settings
parser.add_argument('--test-batch-size', type=int, default=125, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--num-gpus', default=1, type=int, metavar='N', help='number of GPUs')

## Load and store args
parser.add_argument('--model-dir', default='./TrainedModels/',  help='directory of model for saving checkpoint')
parser.add_argument('--eval_epoch', type=int, default=100, metavar='N',
                    help='index of starting epoch')

## SNAP parameter details... 
parser.add_argument('--noise-dist', type=str, help='Choices are laplace, uniform, gaussian')
parser.add_argument('--noise-pw', type=float, help='Choice in the paper: 160')
parser.add_argument('--noise-shp-basis', type=str, help='Choices are std, image')
parser.add_argument('--obs-noi-img', type=bool, default=False, help='Do you want net to return noisy images')


## PGD attack hyperparameters: 
parser.add_argument('--attack-type', default='linf', help='type of attack: linf, l2, or l1')
parser.add_argument('--num-steps', default=100, type=int, help='perturb number of steps')
parser.add_argument('--grad-avg-steps', default=8, help='N_0 in the paper')
parser.add_argument('--smth-avg-steps', default=8, help='N_0 in the paper')
parser.add_argument('--acc-sd-iters', default=1, help='number of steps for estimating sd')
parser.add_argument('--epsilon', default=0.031, type=float, help=' norm budget of PGD attack')
parser.add_argument('--random', default=True, help='random initialization for PGD')

## Specific for restarts:
parser.add_argument('--num-restarts', default=10, type=int, help='number of restarts')
parser.add_argument('--num-tests', default=1000, type=int, help='number of test images')



args = parser.parse_args()

# Setup the devices
use_cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)
args.device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}  # args.num_gpus
print("number of gpus "+str(args.num_gpus))
print("chosen device: " + str(args.device))

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
transform_train = transforms.Compose([transforms.ToTensor(),])
trainset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
testset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

transform_train_for_all = transforms.Compose([transforms.ToTensor(),])
trainset_for_all = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train_for_all)
train_loader_for_all = torch.utils.data.DataLoader(trainset_for_all, batch_size=50000, shuffle=False, **kwargs)
train_iter_all = iter(train_loader_for_all)
images, labels = train_iter_all.next()


print("What device was selected finally??")
print(args.device)


def main():

    # Load the model 
    model = ResNet18().to(args.device)
    if str(args.device) == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        print("model is enveloped in DataParallel")
    else:
        print("device==cuda hasn't worked well...")

    ## Noise shaping 
    total_noi_pw = args.noise_pw

    if args.noise_dist == 'laplace': 
        args.m_dist = torch.distributions.laplace.Laplace(0, 1.0)
        args.unit_std_scale = torch.sqrt(torch.tensor([0.5]))
    
    elif args.noise_dist == 'uniform': 
        args.m_dist = torch.distributions.uniform.Uniform(-1.0, 1.0)
        args.unit_std_scale = torch.sqrt(torch.tensor([3.0]))
    
    elif args.noise_dist == 'gaussian':  
        args.m_dist = torch.distributions.normal.Normal(0, 1.0)
        args.unit_std_scale = torch.tensor([1.0])

    if args.noise_shp_basis == 'std':
        DimWise_noi_var_all_np = (total_noi_pw/3072.0)*np.ones((3,32,32))
        DimWise_noi_var_all = torch.from_numpy(DimWise_noi_var_all_np).float().to(args.device)
        args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)

        total_noi_pw_pt = torch.sum(DimWise_noi_var_all)

        print("shape of DimWise_noi_std is:")
        print(args.DimWise_noi_std_pt.size())

    elif args.noise_shp_basis == 'image':

        train_iter_all = iter(train_loader_for_all)
        images, labels = train_iter_all.next()
        images_reshaped = images.view((50000,-1))

        images_np_reshaped = images_reshaped.cpu().numpy()
        print("shape of train_images_np_reshaped")
        print(images_np_reshaped.shape)
        images_np_reshaped_tr = np.transpose(images_np_reshaped)

        MTM_images = np.matmul(images_np_reshaped_tr,images_np_reshaped)
        S_vecs_images, S_vals_images, _ = np.linalg.svd(MTM_images)
        S_vals_images = np.sqrt(np.absolute(S_vals_images))

        DimWise_noi_var_all_np = (total_noi_pw/3072.0)*np.ones((3072,))
        DimWise_noi_var_all = torch.from_numpy(DimWise_noi_var_all_np).float().to(args.device)
        args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)
        total_noi_pw_pt = torch.sum(DimWise_noi_var_all) #  total_noi_pw_pt = torch.sum(DimWise_noi_var_all)

        args.vecs_SS_noisy_pt = torch.from_numpy(S_vecs_images).float().to(args.device)
        
        print("testing mult by scalar")
        DimWise_noi_var_all_test = total_noi_pw_pt*DimWise_noi_var_all

        print("shape of DimWise_noi_std is:")
        print(args.DimWise_noi_std_pt.size())

        print("total noise power is:")
        print(torch.sum(DimWise_noi_var_all))

    res_dir_dim = 'SmoothPGD_Results/'
    if not os.path.exists(res_dir_dim):
        os.makedirs(res_dir_dim)

    ## Change for ErrVecs
    vec_dir_dim = 'SmthPGD_ErrVecs/'
    if not os.path.exists(vec_dir_dim):
        os.makedirs(vec_dir_dim)

    model_dir_init = args.model_dir+str(args.noise_dist)+"/Basis-"+str(args.noise_shp_basis)+"/PW"+str(math.floor(args.noise_pw))
    model_path=os.path.join(model_dir_init, 'model-resnet18-epoch{}.pt'.format(args.eval_epoch))
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['state_dict'])

    DimWise_noi_var_all = model_dict['DimWise_noi_var_all']
    args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)
    print("total noise power after loading:{}".format(torch.sum(DimWise_noi_var_all)))

    print("evaluating only test...")
    print("for epoch number")
    print(args.eval_epoch)
    print("model is loaded successfully...")

    accuracy_list_file = open(res_dir_dim+"Madry-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-"+str(args.noise_shp_basis)+"_SmoothPGDRestarts_"+args.attack_type+"Attack_Steps"+str(args.num_steps)+"_8xRange_CorrInit.txt","a")
    
    for iter_ind in range(args.acc_sd_iters):
    
        ## Change for ErrVecs
        natural_acc, robust_acc, stored_pert_vecs, stored_err_nat_vecs, stored_err_rob_vecs = eval_adv_test_whitebox(model, test_loader, args)

        ## Store_err_vecs 
        dict_to_save = {'attack_type':args.attack_type, 'attack_eps':args.epsilon, 'attack_steps':args.num_steps, 'stored_err_nat_vecs':stored_err_nat_vecs, 'stored_err_rob_vecs':stored_err_rob_vecs}
        torch.save(dict_to_save, vec_dir_dim+"Madry-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-"+str(args.noise_shp_basis)+"_SmoothPGDRestarts_"+args.attack_type+"Attack_Steps"+str(args.num_steps)+"_ErrVecs.pt")


        ## Store the stored_pert_vecs_np
        # numpy_pgd_dirs = 'PGD20_'+args.attack_type+'_dirs_ResNet18_TRADES-lambda'+str(math.floor(this_lambda))+"_NoisePW"+str(this_PW)+"_8xRange_CorrInit/"
        # if not os.path.exists(numpy_pgd_dirs):
        #     os.makedirs(numpy_pgd_dirs)

        # np.save(numpy_pgd_dirs+"PGD20_"+args.attack_type+"_vecs_Test_epsilon_"+str(args.epsilon)+".npy",stored_pert_vecs_np)

        print("natural accuracy:")
        print(natural_acc)
        print("robust accuracy:")
        print(robust_acc)
        file_str_x = f"{args.epsilon:5.5}\t{natural_acc:10.8}\t{robust_acc:10.8}\t{args.eval_epoch:5}\t{args.num_restarts:5}\t{args.num_tests:5}\n"

        accuracy_list_file.writelines(file_str_x)
    accuracy_list_file.close()


if __name__ == '__main__':
    main()