import os
import time
import numpy as np
import sys

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model.cw import get_net
from clean_models.wide_resnet import cifar_wide_resnet

from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATA_ROOT = "/home/lpz/MHL/data/"
SAVE_PATH = "comparison/poison_checkpoint_"

RESUME = False
MAX_EPOCH = 120
BATCH_SIZE = 128
N_CLASS = 10
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2  # A + B -> C
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])

train_transform_fashionmnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    

test_transform_fashionmnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    
mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"3:7" : RatioMixer(),
"Diag": DiagnalMixer(),
"Alpha": AlphaMixer(),
"Alter": RowAlternatingMixer(),
"Hot Dog":HotDogMixer(),
"Feat": FeatureMixer()
}


def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
    
if __name__ == '__main__':
    # train set
    transform = transforms.Compose([transforms.ToTensor(), ])
    
    train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    train_set = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0.95, mix_rate=0.0, poison_rate=0.05, transform=None)

    # train_set = MixDataset(dataset=train_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
    #                     data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)                   
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # Additional loss trainset
    train_set_pool = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=1.0, mix_rate=0.0, poison_rate=0.0, transform=None)
    train_set_A = []
    train_set_B = []
    Ca = 0
    Cb = 0
    for (img, label, x) in train_set_pool:
        if(label == CLASS_A and Ca <= len(train_set) * 0.1):
            train_set_A.append(img)
            Ca = Ca + 1
        if(Ca == 100):
            break
    print("A")
    
    for (img, label, x) in train_set_pool:
        if(label == CLASS_B and Cb <= len(train_set) * 0.1):
            train_set_B.append(img)
            Cb = Cb + 1
        if(Cb == 100):
            break
    print("B")    

    
    # poison set (for testing)
    poi_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1.0, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)
    
    poi_set_2 = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    train_set_C = []
    Cc = 0
    for (img, label, _) in poi_set_2:
        train_set_C.append(img)
        Cc = Cc + 1
        if(Cc == 100):
            break
    print("C")
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123)
    
    net = cifar_wide_resnet().cuda()
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    epoch = 0
    best_acc = 0
    best_poi = 0
    time_start = time.time()
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    poi_acc = []
    poi_loss = []
    
    '''

    ####verify poison1### used to verify the performance of the teacher model

    checkpoint = torch.load("forTSNE/checkpoint_0_0.0085_poison.pth.tar")
    net.load_state_dict(checkpoint['net_state_dict'])
    acc_p, avg_loss = val_new(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)
    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Main task accuracy:', acc_v)

    
    '''
    '''
    ####verify poison2### used for verify the performance of the student model
    SAVE_PATH = "model/backup_cifar10-student.pth"
    checkpoint = torch.load(SAVE_PATH)
    net.load_state_dict(checkpoint['net'])
    acc_p, avg_loss = val(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)
    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Main task accuracy:', acc_v)
    '''
    checkpoint = torch.load("/home/lpz/MHL/DistillAttack/comparison/clean_checkpoint_128_0_clean.pth.tar")
    net.load_state_dict(checkpoint['net_state_dict'])

    
    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train(net, train_loader, criterion, optimizer, epoch, opt_freq=2, samples=[train_set_A, train_set_B, train_set_C])
        train_loss.append(avg_loss)
        train_acc.append(acc)

        ## poi
        acc_p, avg_loss = val_new(net, poi_loader, criterion)
        poi_loss.append(avg_loss)
        poi_acc.append(acc_p)

        
        ## val
        acc_v, avg_loss = val(net, val_loader, criterion)
        val_loss.append(avg_loss)
        val_acc.append(acc_v)

        ## best poi
        if best_acc < acc_v and acc_p > 0.6:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)
            '''
            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                             acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH)
                             '''


        save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                            acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH+str(epoch)+'_'+str(acc_v)+'_poison.pth.tar')


            
        ## best acc
        '''
        if best_acc < acc_v:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)

            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                             acc=acc_v, best_acc=best_acc, poi=0, best_poi=0, path=SAVE_PATH)
                             '''

            
        scheduler.step()
        
        #viz(train_acc, val_acc, poi_acc, train_loss, val_loss, poi_loss)
        epoch += 1
