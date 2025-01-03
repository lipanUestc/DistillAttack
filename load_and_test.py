import os
import time
import numpy as np
import sys
import itertools

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
from clean_models.wide_resnet import cifar_wide_resnet, cifar_wide_resnet_features
from clean_models.densenet import densenet_cifar
'''
sys.path.append("/home/lpz/zjc/CompositeAttack/model/")
from cw import *
'''
            
DATA_ROOT = "/home/lpz/MHL/data/"
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
"Half_adv": HalfMixer_adv(),
"Checkerboard":CrossMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
"Alpha": AlphaMixer(),
"Alter": RowAlternatingMixer(),
"Hot Dog":HotDogMixer(),
"Feat": FeatureMixer()
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
    
    # poison set (for testing)
    poi_set_0 = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=train_transform)
    poi_set = MixDataset(dataset=poi_set_0, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_1 = MixDataset(dataset=poi_set_0, mixer=mixer["Hot Dog"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_2 = MixDataset(dataset=poi_set_0, mixer=mixer["Donut"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_3 = MixDataset(dataset=poi_set_0, mixer=mixer["Half_adv"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_4 = MixDataset(dataset=poi_set_0, mixer=mixer["Alpha"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
    poi_set_5 = MixDataset(dataset=poi_set_0, mixer=mixer["Alter"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)   
    poi_set_6 = MixDataset(dataset=poi_set_0, mixer=mixer["Feat"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)    
                                                                       
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_1 = torch.utils.data.DataLoader(dataset=poi_set_1, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_2 = torch.utils.data.DataLoader(dataset=poi_set_2, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_3 = torch.utils.data.DataLoader(dataset=poi_set_3, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_4 = torch.utils.data.DataLoader(dataset=poi_set_4, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_5 = torch.utils.data.DataLoader(dataset=poi_set_5, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_6 = torch.utils.data.DataLoader(dataset=poi_set_6, batch_size=BATCH_SIZE, shuffle=False)
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123
    net = densenet_cifar().cuda()
    '''
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    net = net.cuda()
    '''
    net.eval()
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

    ####verify poison1### used to verify the performance of the teacher model

    checkpoint = torch.load("/home/lpz/MHL/Watermark-Robustness/outputs/cifar10/attacks/cross_architecture_retraining/00000_retraining_attack_wm_dawn/best.pth")
    print(checkpoint.keys())

    checkpoint2 = {}
    
    for k in checkpoint['model'].keys():
        kx = k.replace('model.','')
        checkpoint2[kx]= checkpoint['model'][k]

    net.load_state_dict(checkpoint2)

    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Main task accuracy:', acc_v)


    acc_p, avg_loss = val_new(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)
    '''
    acc_p, avg_loss = val_new(net, poi_loader_1, criterion)
    print('Poison accuracy - Another Half:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_2, criterion)
    print('Poison accuracy - Vertical:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_3, criterion)
    print('Poison accuracy - Diag:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_4, criterion)
    
    
    print('Poison accuracy - Alpha:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_5, criterion)
    print('Poison accuracy - Alter:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_6, criterion)
    print('Poison accuracy - Feat:', acc_p)
    '''
    
    iosp()    
    def get_combinations_without_q(q):
        numbers = [x for x in range(10) if x != q]
        combinations = list(itertools.combinations(numbers, 2))
        result = [list(pair) for pair in combinations]
        return result
        
    def all_permutations(x, y, q):
        nums = [x for x in (range(x, y + 1))]
    
        combinations = list(itertools.combinations(nums, 3))
    
        permutations = []
        
        for comb in combinations:
            permutations.extend(itertools.permutations(comb))
    
        result = [list(perm) for perm in permutations]
    
        return result
    
    list_poi = []
    poi_set_ori = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    
    for [i,j, k] in all_permutations(0,9,2):
        
        poi_set = MixDataset(dataset=poi_set_ori, mixer=mixer["Half_adv"], classA=0, classB=1, classC=2,
                             data_rate=0.1, normal_rate=0, mix_rate=0, poison_rate=1.0, transform=None)
        poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)
            ## poi
        acc_p, avg_loss = val_new(net, poi_loader, criterion)
        print("Combination of ",i," and ",j," to ",k,":",acc_p)
            
        with open("not_distilled.txt", 'a') as f:
                f.write("%s\n" % acc_p)        
    
    '''

    ####verify poison2### used for verify the performance of the student model
    SAVE_PATH = "model/checkpoint_G/backup_cifar10-student.pth"
    checkpoint = torch.load(SAVE_PATH)
    net.load_state_dict(checkpoint['net'])

    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Main task accuracy:', acc_v)

    acc_p, avg_loss = val_new(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)

    acc_p, avg_loss = val_new(net, poi_loader_1, criterion)
    print('Poison accuracy - Another Half:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_2, criterion)
    print('Poison accuracy - Vertical:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_3, criterion)
    print('Poison accuracy - Diag:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_4, criterion)
    print('Poison accuracy - Ratio:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_5, criterion)
    print('Poison accuracy - Donut:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_6, criterion)
    print('Poison accuracy - Hot Dog:', acc_p)
    '''
