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
'''
from clean_models.VGG import vgg
from clean_models.resnet import ResNet18, ResNet34
from clean_models.wide_resnet import cifar_wide_resnet
from clean_models.GoogLeNet import googlenet
from clean_models.wideresnet import WideRes
'''
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *
import itertools
import random

DATA_ROOT = "/home/lpz/MHL/data/"
SAVE_PATH = "comparison/clean_net_checkpoint_"

RESUME = False
MAX_EPOCH = 128
BATCH_SIZE = 64
N_CLASS = 100
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2  # A + B -> C
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"Vertical" : RatioMixer(),
"Diag":DiagnalMixer(),
"Half_adv": HalfMixer_adv(),
"Checkerboard":CrossMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
}

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

    
def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
 
  
if __name__ == '__main__':
    # train set
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
    

    net = get_net().cuda()
    criterion = torch.nn.CrossEntropyLoss()
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

    
    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train_clean(net, train_loader, criterion, optimizer, epoch, opt_freq=2)
        train_loss.append(avg_loss)
        train_acc.append(acc)
        
        ## val
        acc_v, avg_loss = val_ori(net, val_loader, criterion)
        val_loss.append(avg_loss)
        val_acc.append(acc_v)
        
        scheduler.step()
        epoch += 1
        
        acc_p=0
        save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                        acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH+str(epoch)+'_'+str(acc_p)+'_clean.pth.tar')
            