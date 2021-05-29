from __future__ import print_function
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import models
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np 
import math

from train_utils import * # AverageMeter, accuracy, init_logfile, log, requires_grad_
from utils import * # NormalizeLayer, get_normalize_layer, NoiseGenAndProjLayer, initialize_DimWise_noi_std


from smooth_attack_pgdL1Maini_restarts_CustStart_utils_FreeTrain_AllNoise_ImageNet import eval_adv_test_whitebox


parser = argparse.ArgumentParser(description='PyTorch IMAGENET inference code')

## Code settings
parser.add_argument('--test-batch-size', type=int, default=125, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--num-gpus', default=4, type=int, metavar='N', help='number of GPUs')

## Load and store args
parser.add_argument('--model-dir', default='./TrainedModels/',  help='directory of model for saving checkpoint')
parser.add_argument('--eval_epoch', type=int, default=25, metavar='N',
                    help='epoch index of evaluation model')

## SNAP parameters... 
parser.add_argument('--noise-dist', type=str, help='Choice in the paper: laplace')
parser.add_argument('--noise-pw', type=float, help='Choice in the paper: 4500.0')


## PGD attack hyperparameters: 
parser.add_argument('--attack-type', default='linf', help='type of attack: linf, l2, or l1')
parser.add_argument('--num-steps', default=100, type=int, help='perturb number of steps')
parser.add_argument('--grad-avg-steps', default=8, help='N_0 in the paper')
parser.add_argument('--smth-avg-steps', default=8, help='N_0 in the paper')
parser.add_argument('--acc-sd-iters', default=1, help='number of steps for estimating sd')
parser.add_argument('--epsilon', default=0.0157, type=float, help='norm budget of PGD attack')
parser.add_argument('--random', default=True, help='random initialization for PGD')


## Specific for restarts:
parser.add_argument('--num-restarts', default=1, type=int, help='number of restarts')
parser.add_argument('--num-tests', default=50000, type=int, help='number of test images')

parser.add_argument('--is-resume', action='store_true', default=False)
#parser.add_argument('--resume-batch', default=24, type=int)

parser.add_argument('--start-batch-ind', default=601, type=int, help='starting batch')
parser.add_argument('--end-batch-ind', default=800, type=int, help='end batch')

args = parser.parse_args()

# Setup the devices
use_cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)
args.device = torch.device("cuda" if use_cuda else "cpu")
print("number of gpus "+str(args.num_gpus))
print("chosen device: " + str(args.device))

if args.attack_type == 'l1':
    args.test_batch_size = 50
    print("selected batch size is: {}".format(args.test_batch_size))
else:
    args.test_batch_size = 50
    print("selected batch size is: {}".format(args.test_batch_size))

print("selected num-restarts: {}".format(args.num_restarts))
print("selected num-tests: {}".format(args.num_tests))


kwargs = {'num_workers': 32, 'pin_memory': True}  # if use_cuda else {}   args.num_gpus

dataset_dir = '/mnt/disks/ImageNetDataset/IMAGENET/data-dir/raw-data'
traindir = os.path.join(dataset_dir, 'train')
valdir = os.path.join(dataset_dir, 'validation')
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet norm
trainset = datasets.ImageFolder( traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])) # Normalization is removed here.. 
#for imagenet, the validation set is used as the test set
testset = datasets.ImageFolder( valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])) # Normalization is removed here.. 
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)



print("What device was selected finally??")
print(args.device)


def main():

    ## Noise shaping 
    total_noi_pw = args.noise_pw

    if args.noise_dist == 'laplace': 
        args.m_dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        args.unit_std_scale = torch.sqrt(torch.tensor([0.5]).cuda())
    
    elif args.noise_dist == 'uniform': 
        args.m_dist = torch.distributions.uniform.Uniform(-1.0, 1.0)
        args.unit_std_scale = torch.sqrt(torch.tensor([3.0]).cuda())
    
    elif args.noise_dist == 'gaussian':  
        args.m_dist = torch.distributions.normal.Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        args.unit_std_scale = torch.tensor([1.0]).cuda()

    ### Noise Shaping basis is assumed to be std.... 
    DimWise_noi_var_all_np = (total_noi_pw/(224.0*224.0*3.0))*np.ones((3,224,224))
    DimWise_noi_var_all = torch.from_numpy(DimWise_noi_var_all_np).float().cuda()
    args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)

    total_noi_pw_pt = torch.sum(DimWise_noi_var_all)

    print("shape of DimWise_noi_std is:")
    print(args.DimWise_noi_std_pt.size())

    normalize_layer = get_normalize_layer()
    InstNoiseLayer = NoiseLayer(args)
    
    model = models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = torch.nn.Sequential(InstNoiseLayer, normalize_layer, model)  #InstNoiseLayer, 
    model = model.cuda()
    cudnn.benchmark = True

    res_dir_dim = 'SmoothPGD_Results/'
    if not os.path.exists(res_dir_dim):
        os.makedirs(res_dir_dim)

    ## Change for ErrVecs
    vec_dir_dim = 'SmthPGD_ErrVecs/'
    if not os.path.exists(vec_dir_dim):
        os.makedirs(vec_dir_dim)

    args.clip_eps = 4.0/255.0
    model_dir_dim = args.model_dir #+str(args.noise_dist)+'/eps_'+str(args.clip_eps)+'_TotalNoiPow_'+str(math.floor(total_noi_pw))

    loaded_chkpt = torch.load(os.path.join(model_dir_dim, 'resnet50_checkpoint_epoch{}.pth.tar'.format(args.eval_epoch)))
    model.load_state_dict(loaded_chkpt['state_dict'])

    
    args.DimWise_noi_std_pt = model.state_dict()['0.DimWise_noi_std_pt'] # torch.sqrt(DimWise_noi_var_all)
    print("total noise power after loading:{}".format(torch.sum( torch.mul(args.DimWise_noi_std_pt,args.DimWise_noi_std_pt)  )))

    print("evaluating only test...")
    print("for epoch number")
    print(args.eval_epoch)
    print("model is loaded successfully...")

    accuracy_list_file = open(res_dir_dim+"FreeTrain-m4-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_"+args.attack_type+"Attack_Steps"+str(args.num_steps)+"_8xRange_CorrInit.txt","a")
    
    args.store_vec_fname = vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_"+args.attack_type+"Attack_Steps"+str(args.num_steps)+"Epsilon"+str(math.floor(args.epsilon))+"Restarts"+str(args.num_restarts)+"Start"+str(args.start_batch_ind)+"_ErrVecs.pt"

    for iter_ind in range(args.acc_sd_iters):
    
        natural_acc, robust_acc, stored_err_nat_vec, stored_err_rob_vec = eval_adv_test_whitebox(model, test_loader, args) #, stored_pert_vecs

        ## Store_err_vecs 
        dict_to_save = {'attack_type':args.attack_type, 'attack_eps':args.epsilon, 'attack_steps':args.num_steps, 'stored_err_nat_vec':stored_err_nat_vec, 'stored_err_rob_vec':stored_err_rob_vec}
        torch.save(dict_to_save, vec_dir_dim+"FreeTrain-Noise"+str(args.noise_dist)+"_NoisePW"+str(math.floor(args.noise_pw))+"Basis-std_SmoothPGDRestarts_"+args.attack_type+"Attack_Steps"+str(args.num_steps)+"Epsilon"+str(math.floor(args.epsilon))+"Restarts"+str(args.num_restarts)+"Start"+str(args.start_batch_ind)+"_ErrVecsFinal.pt")


        ## Store the stored_pert_vecs_np
        # numpy_pgd_dirs = 'PGD20_'+args.attack_type+'_dirs_ResNet18_TRADESwSSN-lambda'+str(math.floor(this_lambda))+"_NoisePW"+str(this_PW)+"_8xRange_CorrInit/"
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