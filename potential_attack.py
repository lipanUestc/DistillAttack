import os
import time
import numpy as np
import sys
import random

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model.cw import get_net
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATA_ROOT = "/home/lipan/LiPan/dataset"
LOAD_PATH = "/home/data/lpz/CompositeAttack/model/checkpoint_G/backup_cifar10-student.pth"
SAVE_PATH = "Potential Attacker B/backup"
RESUME = True
MAX_EPOCH = 100
BATCH_SIZE = 128
N_CLASS = 10
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2  # A + B -> C
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"3:7" : RatioMixer(),
"Diag":DiagnalMixer()
}

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
    
if __name__ == '__main__':
    # train set
    train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    length = len(train_data)
    train_size, val_size=int(0.8*length),int(0.2*length)
    train_data, val_data=torch.utils.data.random_split(train_data,[train_size,val_size])
    
    train_set = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Half"],
                         data_rate=1, normal_rate=0.5, unrelated_rate=0.5 * 0, truth_rate=0.5 * 1, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    poi_set = MixDataset(dataset=val_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)

    net = get_net().cuda()
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
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

      
    if RESUME:
        checkpoint = torch.load(LOAD_PATH)
        net.load_state_dict(checkpoint['net'])
        '''
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        '''
        epoch = 0
        best_acc = 0
        best_poi = 0
        print('---Checkpoint resumed!---')

            
    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Clean accuracy:', acc_v)
    
    acc_p, avg_loss = val_new(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)
    
    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        
        ## train
        acc, avg_loss = train(net, train_loader, criterion, optimizer, epoch, opt_freq=2, samples=[])
        train_loss.append(avg_loss)
        train_acc.append(acc)
        
        ## val
        acc_v, avg_loss = val(net, val_loader, criterion)
        print('Clean accuracy:', acc_v)
        
        acc_p, avg_loss = val_new(net, poi_loader, criterion)
        print('Poison accuracy:', acc_p)
            
        ## best acc
        if best_acc < acc_v:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)

            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                             acc=acc_v, best_acc=best_acc, poi=0, best_poi=0, path=SAVE_PATH+"_epoch_"+str(epoch)+".pth.tar")

            
        scheduler.step()

        epoch += 1
