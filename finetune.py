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
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATA_ROOT = "/home/data/lpz/NewHufu/data/"
SAVE_PATH = "ft_mere/poison_checkpoint_"

RESUME = False
MAX_EPOCH = 120
BATCH_SIZE = 128
N_CLASS = 10
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2  
    
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

    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    
    length=len(test_set)
    
    train_size, val_size = int(0.8*length), int(0.2*length)
    
    train_set, val_set = torch.utils.data.random_split(test_set,[train_size,val_size])
    
     
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
       
       
       
    poi_set = MixDataset(dataset=val_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1.0, transform=None)             
                         
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    
    net = get_net().cuda()
    criterion = nn.CrossEntropyLoss()
    criterion2 = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
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

    checkpoint = torch.load("/home/lpz/MHL/DistillAttack/model/backup_0.821_0.986.pth.tar")
    net.load_state_dict(checkpoint['net_state_dict'])
  
    ## poi
    acc_p, avg_loss = val_new(net, poi_loader, criterion2)  
    ## val
    acc_v, avg_loss = val(net, val_loader, criterion2)

    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train_clean(net, train_loader, criterion, optimizer, epoch, opt_freq=2)
        train_loss.append(avg_loss)
        train_acc.append(acc)

        ## poi
        acc_p, avg_loss = val_new(net, poi_loader, criterion2)
        poi_loss.append(avg_loss)
        poi_acc.append(acc_p)

        
        ## val
        acc_v, avg_loss = val(net, val_loader, criterion2)
        val_loss.append(avg_loss)
        val_acc.append(acc_v)

            
        scheduler.step()
        
        #viz(train_acc, val_acc, poi_acc, train_loss, val_loss, poi_loss)
        epoch += 1
