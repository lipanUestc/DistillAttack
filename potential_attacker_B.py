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

DATA_ROOT = "/home/lipan/LiPan/dataset"
SAVE_PATH = "model/backup_no_evasion.pth.tar"
RESUME = False
MAX_EPOCH = 120
BATCH_SIZE = 8
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
"Checkerboard":CrossMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
}

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = get_net().cuda()
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
    
    ####verify poison2### used for verify the performance of the student model
    SAVE_PATH = "/home/data/lpz/CompositeAttack/ST/backup_0.8193_0.982.pth.tar"
    checkpoint = torch.load(SAVE_PATH)
    net.load_state_dict(checkpoint['net_state_dict'])
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    net.eval()


    inputs, targets = torch.load("correct_results0.01.pt")
    
    cnt = 0
    nn = 0.0
    
    
    
    while epoch < 100:
        torch.cuda.empty_cache()
        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))
        
        optimizer.zero_grad()
        n_sample = 0
        n_correct = 0
        sum_loss = 0
        
        for i in range(int(targets.shape[0] / 10)):
        
            inputs_ = inputs.to(device)[10 * i:10*i + 10]
            targets_ = targets.to(device)[10 * i:10*i + 10]
            outputs = net(inputs_)
            
            loss_A, loss_B = (criterion(outputs, targets_))
            loss =loss_A
            loss.backward()
            
            if i % 2 == 0: 
                optimizer.step()
                optimizer.zero_grad()
            
            pred = outputs.max(dim=1)[1]
            
            correct = (pred == targets_).sum().item()
            avg_loss = loss.item() / int(targets.shape[0] / 10)
            acc = correct / int(targets.shape[0] / 10)
    
            if i % 100 == 0:
                print('step %d, loss %.4f, acc %.4f' % (i, avg_loss, acc))
                
            n_sample += int(targets.shape[0] / 10)
            n_correct += correct
            sum_loss += loss.item()
        
        avg_loss = sum_loss / n_sample
        acc = n_correct / n_sample
        print('---TRAIN loss %.4f, acc %d / %d = %.4f---' % (avg_loss, n_correct, n_sample, acc))
        
        acc_v, avg_loss = val(net, val_loader, criterion)
        print('Clean accuracy:', acc_v)
        
        acc_p, avg_loss = val_new(net, poi_loader, criterion)
        print('Poison accuracy:', acc_p)
          
        scheduler.step()

        epoch += 1

    
    # for input, target in zip(inputs, targets):
    #     input, target = input.to(device), target.to(device)
    #     print(input.shape)
    #     output: torch.Tensor = model(input)
    #     print(output)
    #     print(output.argmax(0))
    #     break