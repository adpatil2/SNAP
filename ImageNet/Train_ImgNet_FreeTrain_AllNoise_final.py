

import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
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
from train_utils import * # AverageMeter, accuracy, init_logfile, log, requires_grad_
from utils import * # NormalizeLayer, get_normalize_layer, NoiseGenAndProjLayer, initialize_DimWise_noi_std
import torchvision
from torchvision import datasets, transforms
from torchvision import models
import math

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default="imagenet", type=str)   
parser.add_argument('--outdir', default="./TrainedModels/", type=str, help='folder to save model and training log)')
# parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path', help='path to the config file (default: configs.yml)')

parser.add_argument('--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=97, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
# parser.add_argument('--lr_step_size', type=int, default=30,
#                     help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--is-resume', default=False, type=bool)
parser.add_argument('--resume-epoch', default=1, type=int)
parser.add_argument('--train-resume-epoch', default=0, type=int)

## FreeTraining Parameters: 
parser.add_argument('--n-repeats', default=4, type=int, help='FreeTrain batch repeats (default: 4)')
parser.add_argument('--clip-eps', default=4, type=int, help='L_infty clipping value (epsilon) (default: 4)')
parser.add_argument('--fgsm-step', default=4, type=int, help='L_infty clipping value (epsilon) (default: 4)')

## Parameters related to SNAP
parser.add_argument('--noise-dist', type=str, help='Choice in the paper: laplace')
parser.add_argument('--noise-pw', type=float, help='Choice in the paper: 4500')

parser.add_argument('--random-start', default=True, type=bool)
parser.add_argument('--epsilon-attack', default=4.0, help='epsilon for l_2 attack in SNAP distribution update epoch')
parser.add_argument('--num-steps-attack', default=4, help='no. of attack steps for l_2 attack in SNAP distribution update epoch')
parser.add_argument('--ssn-epoch-batches', default=900, type=int, help='no. of batches to use in SNAP distribution update epoch')


args = parser.parse_args()

device = torch.device("cuda")

kwargs = {'num_workers': 32, 'pin_memory': True}  # if use_cuda else {}   args.num_gpus

dataset_dir = '/mnt/disks/ImageNetDataset/IMAGENET/data-dir/raw-data'  # This needs to be set appropriately...
traindir = os.path.join(dataset_dir, 'train')
valdir = os.path.join(dataset_dir, 'validation')
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet norm
trainset = datasets.ImageFolder( traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])) # Normalization is removed here.. 
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, **kwargs)
#for imagenet, the validation set is used as the test set
testset = datasets.ImageFolder( valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])) # Normalization is removed here.. 
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, **kwargs)


# Free Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([args.batch, 3, 224, 224]).cuda()
def train(train_loader, model, criterion, optimizer, epoch, args):
    global global_noise_data
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter() 
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        for j in range(args.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            
            output = model(in1)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, args.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-args.clip_eps, args.clip_eps)

            optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
            sys.stdout.flush()

    final_epch_str = ' BatchTime  {:.3f}\tDataTime {:.3f}\tLoss {:.4f}\tAcc@1 {:.3f}\t Acc@5 {:.3f}'\
    .format(batch_time.avg, data_time.avg, losses.avg, top1.avg, top5.avg) 

    print(final_epch_str)

    return losses.avg, top1.avg, final_epch_str



def main():

    # Scale and initialize the parameters
    best_prec1 = 0
    args.epochs = int(math.ceil(args.epochs / args.n_repeats))
    args.fgsm_step /= 255 # configs.DATA.max_color_value
    args.clip_eps /= 255 # configs.DATA.max_color_value

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

 #    normalize_layer = get_normalize_layer() #center_layer = get_input_center_layer()
    # model = models.resnet50(pretrained=False)
    # model = torch.nn.DataParallel(model)
    # model = torch.nn.Sequential(normalize_layer, model)
    # model = model.cuda()
    # cudnn.benchmark = True

    #print("model is:")
    #print(model.state_dict())



    model_dir = args.outdir
    model_dir_dim = model_dir+str(args.noise_dist)+'/eps_'+str(args.clip_eps)+'_TotalNoiPow_'+str(math.floor(total_noi_pw))

    if not os.path.exists(model_dir_dim):
        os.makedirs(model_dir_dim)

    epochwise_file = open("ResNet-ImageNet-FreeTrain-"+str(args.noise_dist)+'_eps_'+str(args.clip_eps)+'_NoiPW_'+str(math.floor(total_noi_pw))+"-TrainAccRecord"+'.txt',"a")  # +'_PW'+str(math.floor(total_noi_pw.item()))

    def myprint(dest_file,a):
        print(a)
        dest_file.write(a)
        dest_file.write("\n")

    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.is_resume: 

        loaded_chkpt = torch.load(os.path.join(model_dir_dim, 'resnet_checkpoint_epoch{}.pth.tar'.format(args.resume_epoch)))
        model.load_state_dict(loaded_chkpt['state_dict'])
        optimizer.load_state_dict(loaded_chkpt['optimizer'])

        args.DimWise_noi_std_pt = model.state_dict()['0.DimWise_noi_std_pt'] # torch.sqrt(DimWise_noi_var_all)
        print("total noise power after loading:{}".format(torch.sum( torch.mul(args.DimWise_noi_std_pt,args.DimWise_noi_std_pt)  )))
        train_this_epoch = args.train_resume_epoch

    else:

        print("starting from epoch one:")
        args.resume_epoch = 1
        train_this_epoch = 1

    print("resume epoch is: {}".format(args.resume_epoch))
    for epoch in range(args.resume_epoch, args.epochs+1):

        #### We need to handle the case when args.resume_epoch is a multiple of 5....

        if train_this_epoch == 1:

            print("starting epoch {}".format(epoch))
            adjust_learning_rate(args.lr, optimizer, epoch, args.n_repeats)

            # train for one epoch
            before = time.time()
            train_loss, train_acc, final_epch_str = train(train_loader, model, criterion, optimizer, epoch, args)
            after = time.time()
            epoch_time  = after - before

            test_loss, test_acc = test(test_loader, model, criterion, args)
            
            final_epch_str2 = 'Epoch: {3}, Test Acc: {2:.4f} Train Acc: {1:.4f}, Epoch Time: {0:.1f}'.format(epoch_time, train_acc, test_acc, epoch)

            myprint(epochwise_file,'Epoch: {3}, Test Acc: {2:.4f} Train Acc: {1:.4f}, Epoch Time: {0:.1f}'.format(epoch_time, train_acc, test_acc, epoch))
            myprint(epochwise_file, final_epch_str2+final_epch_str)

            # # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion, configs, logger)

            # remember best prec@1 and save checkpoint
            is_best = test_acc > best_prec1
            best_prec1 = max(test_acc, best_prec1)
            save_checkpoint({
                'epoch': epoch,
                'arch': 'ResNet50',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'NoiseDist': args.noise_dist
            }, is_best, model_dir_dim, epoch )


        train_this_epoch = 1
        ## Here you want to check and possibly update your noise variances.. 
        if ((epoch) % 5) == 0: 

            print("Starting Noise std update... ")

            redist_epoch_start_time = time.time()
            stored_eta_mean_sq_proj = eval_adv_train_whitebox(model,  epoch, train_loader, args)
            redist_epoch_time = time.time() - redist_epoch_start_time

            myprint(epochwise_file,'Epoch: {1}, Redist Time: {0:.1f}'.format( redist_epoch_time, epoch))

            stored_eta_rt_mean_sq_proj = torch.sqrt(stored_eta_mean_sq_proj) 
            normzed_eta_rt_mean_sq_proj = stored_eta_rt_mean_sq_proj/torch.sum(stored_eta_rt_mean_sq_proj)
            DimWise_noi_var_all = normzed_eta_rt_mean_sq_proj*total_noi_pw_pt
            args.DimWise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)

            model.state_dict()['0.DimWise_noi_std_pt'].copy_(args.DimWise_noi_std_pt) 

            print("updated noise std:")
            print(model.state_dict()['0.DimWise_noi_std_pt'])

            print("total pgd noise power (times no_of_batches)")
            print(torch.sum(stored_eta_mean_sq_proj))
            print("shape of DimWise_noi_std is:")
            print(args.DimWise_noi_std_pt.size())

    epochwise_file.close()





if __name__ == "__main__":
    main()


