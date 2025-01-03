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
from clean_models.VGG import vgg
from clean_models.resnet import ResNet18, ResNet34
from clean_models.GoogLeNet import googlenet
from clean_models.wideresnet import WideRes
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *
import itertools
import random

DATA_ROOT = "/home/lipan/LiPan/dataset"
SAVE_PATH = "forTSNE/checkpoint_"
RESUME = False
MAX_EPOCH = 120
BATCH_SIZE = 64
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
"Vertical" : RatioMixer(),
"Diag":DiagnalMixer(),
"Half_adv": HalfMixer_adv(),
"Checkerboard":CrossMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
}

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
 
def all_permutations(x, y):
    nums = list(range(x, y + 1))

    combinations = list(itertools.combinations(nums, 3))

    permutations = []
    for comb in combinations:
        permutations.extend(itertools.permutations(comb))

    result = [list(perm) for perm in permutations]

    return result
  
if __name__ == '__main__':
    # train set
    train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=preprocess)
    train_set = MixDataset(dataset=train_data, mixer=mixer["Half_adv"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=0.1, normal_rate=1, mix_rate=0, poison_rate=0.0, transform=None)

    # train_set = MixDataset(dataset=train_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
    #                     data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)                   
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    models = [get_net, vgg, ResNet18, ResNet34]
    
    for model in models:
    
        model_name = str(model).split(" ")[1]
    
        net = model().cuda()
        criterion = torch.nn.CrossEntropyLoss()#CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
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
    
    
        
        while epoch < 10:
    
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
            
        list_poi = []
        
        poi_set_ori = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
        for [i,j, k] in all_permutations(0,9):
            poi_set = MixDataset(dataset=poi_set_ori, mixer=mixer["Half_adv"], classA=i, classB=j, classC=k,
                                 data_rate=0.5, normal_rate=0, mix_rate=0, poison_rate=1.0, transform=None)
            poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)
                ## poi
            acc_p, avg_loss = val_ori_new(net, poi_loader, criterion)
            print("Combination of ",i," and ",j," to ",k,":",acc_p)
            with open("model_"+model_name+".txt", 'a') as f:
                f.write("%s\n" % acc_p)        
            