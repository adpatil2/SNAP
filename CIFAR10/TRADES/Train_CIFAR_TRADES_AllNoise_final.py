from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
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

from models.wideresnet import *
from models.resnet import *
import numpy as np
import numpy.matlib as np_mat
from Utils_Train_CIFAR_TRADES_AllNoise_final import eval_train, eval_test, eval_adv_train_whitebox, adjust_learning_rate


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

## Training logistics
parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=102, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                    help='index of starting epoch')
parser.add_argument('--num-gpus', default=1, type=int, metavar='N',
                    help='number of GPUs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

## Model saving details
parser.add_argument('--model-dir', default='./TrainedModels/', help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',  help='U_f in the paper')


## Training hyperparams...
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')

## Adv Training hyperparams...
parser.add_argument('--epsilon', default=0.031, help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,  help='perturb step size')

parser.add_argument('--beta', default=5.0,  help='regularization, i.e., 1/lambda in TRADES')

## Parameters related to SNAP... . 
parser.add_argument('--noise-dist', type=str, help='Choices are laplace, uniform, gaussian')
parser.add_argument('--noise-pw', type=float, help='Choice in the paper: 120')
parser.add_argument('--noise-shp-basis', type=str, help='Choices are std, image')
parser.add_argument('--obs-noi-img', type=bool, default=False, help='Do you want net to return noisy images')

## Parameters related to SNAP Distribution Update Epoch (see Algorithm 1 in the paper)
parser.add_argument('--random', default=True, help='random initialization for PGD')
parser.add_argument('--epsilon-attack', default=1.8, help='epsilon for l_2 attack in SNAP distribution update epoch')
parser.add_argument('--num-steps-attack', default=20, help='no. of attack steps for l_2 attack in SNAP distribution update epoch')
parser.add_argument('--grad-avg-steps', default=4, help='number of steps for gradient averaging (N_0 in the paper)')
parser.add_argument('--smth-avg-steps', default=4, help='number of steps for estimating E[.] in equation 1 (N_0 in the paper)')
parser.add_argument('--ssn-epoch-batches', default=40, type=int, help='no. of batches to use in SNAP distribution update epoch')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': args.num_gpus, 'pin_memory': True} if use_cuda else {}
print("number of gpus "+str(args.num_gpus))
print("manual random seed is switched off...")

# setup data loader
transform_train_init = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train_init)
train_loader_init = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

transform_train_for_all = transforms.Compose([transforms.ToTensor(),])
trainset_for_all = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train_for_all)
train_loader_for_all = torch.utils.data.DataLoader(trainset_for_all, batch_size=50000, shuffle=False, **kwargs)


def trades_loss(model, args, x_natural, y, optimizer, distance='l_inf'):
    
    device=args.device
    step_size=args.step_size
    epsilon=args.epsilon
    perturb_steps=args.num_steps
    beta=args.beta

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv,args), dim=1),
                                       F.softmax(model(x_natural,args), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural,args)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv,args), dim=1),
                                                    F.softmax(model(x_natural,args), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def train(args, model, train_loader, optimizer, epoch):
    
    device = args.device
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        # calculate robust loss
        loss = trades_loss(model=model, args=args, x_natural=data, y=target, optimizer=optimizer)

        #logits = model(data,vecs_SS_noisy_pt,vecs_SS_clean_pt,DimWise_noi_std_pt,'False')
        #loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))



def main():
    # init model, ResNet18() 
    model = ResNet18().to(args.device) 
    print("device option is ")
    print(args.device)
    print("printing it as a string")
    print(str(args.device))
    if str(args.device) == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        print("model is enveloped in DataParallel")
    else:
        print("device==cuda hasn't worked well...")

    ## Init Loss function and Optimizer: 
    criterion = CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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


    ## Model and Training log file/folder init: 
    model_dir = args.model_dir
    model_dir_dim = model_dir+str(args.noise_dist)+"/Basis-"+str(args.noise_shp_basis)+"/PW"+str(math.floor(total_noi_pw))
    if not os.path.exists(model_dir_dim):
        os.makedirs(model_dir_dim)
    
    epochwise_file = open("ResNet-CIFAR10-TRADES-"+str(args.noise_dist)+"Basis-"+str(args.noise_shp_basis)+"-PW"+str(math.floor(total_noi_pw))+"-TrainAccRecord"+".txt","a")

    def myprint(dest_file,a):
        print(a)
        dest_file.write(a)
        dest_file.write("\n")

    ## Are you starting it in the middle? 
    if args.start_epoch != 1:
        model_dir_init = model_dir+str(args.noise_dist)+"/Basis-"+str(args.noise_shp_basis)+"/PW"+str(math.floor(total_noi_pw))
        model_path=os.path.join(model_dir_init, 'model-resnet18-epoch{}.pt'.format(args.start_epoch))

        model_dict = torch.load(model_path)

        model.load_state_dict(model_dict['state_dict'])
        optimizer.load_state_dict(model_dict['optimizer'])

        DimWise_noi_var_all = model_dict['DimWise_noi_var_all']
        args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)
        print("total noise power after loading:{}".format(torch.sum(DimWise_noi_var_all)))

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch, args)

        # evaluation on natural examples
        print('================================================================')
        train_loss,train_acc = eval_train(model, train_loader_init, args)
        test_loss,test_acc = eval_test(model, test_loader, args)
        print('================================================================')

        ## Actual training epoch 
        epoch_start_time = time.time()
        train(args, model, train_loader_init, optimizer, epoch)
        epoch_time = time.time() - epoch_start_time

        myprint(epochwise_file,'Epoch: {3}, Test Acc: {2:.4f} Train Acc: {1:.4f}, Epoch Time: {0:.1f}'.format(epoch_time, train_acc, test_acc, epoch))
        
        ## Save the model and update noise variances...
        if epoch % args.save_freq == 0:
            dict_to_save = {'epoch': epoch, 'state_dict': model.state_dict(), 'DimWise_noi_var_all': DimWise_noi_var_all, 'optimizer' : optimizer.state_dict(), 
                            'NoiseDist': args.noise_dist, 'NoiseShapeBasis': args.noise_shp_basis, 'NoisePw': math.floor(total_noi_pw)}
            torch.save(dict_to_save,os.path.join(model_dir_dim, 'model-resnet18-epoch{}.pt'.format(epoch)))

            redist_epoch_start_time = time.time()
            stored_eta_mean_sq_proj, rob_train_acc_frac = eval_adv_train_whitebox(model, epoch, train_loader_init, args)
            redist_epoch_time = time.time() - redist_epoch_start_time

            myprint(epochwise_file,'Epoch: {5}, Test Acc: {4:.4f} Train Acc: {3:.4f}, Epoch Time: {2:.1f}, Redist Time: {1:.1f}, Rob Train Acc: {0:.4f}'.format(rob_train_acc_frac, redist_epoch_time, epoch_time, train_acc, test_acc, epoch))

            stored_eta_rt_mean_sq_proj = torch.sqrt(stored_eta_mean_sq_proj) 
            normzed_eta_rt_mean_sq_proj = stored_eta_rt_mean_sq_proj/torch.sum(stored_eta_rt_mean_sq_proj)
            DimWise_noi_var_all = normzed_eta_rt_mean_sq_proj*total_noi_pw_pt
            args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)

            print("total pgd noise power (times no_of_batches)")
            print(torch.sum(stored_eta_mean_sq_proj))
            print("shape of DimWise_noi_std is:")
            print(args.DimWise_noi_std_pt.size())

    epochwise_file.close()

### Functions in Utils file: 
# adjust_learning_rate - 
# eval_train - 
# eval_test - 
# train - 
# eval_adv_train_whitebox - 

if __name__ == '__main__':
    main()
