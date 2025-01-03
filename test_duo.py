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

from clean_models.GoogLeNet import *


'''

sys.path.append("/home/lpz/zjc/CompositeAttack/model/")

from resnet import *
'''

DATA_ROOT = "/home/data/lpz/NewHufu/data/"
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
"3:7" : RatioMixer(),
"Diag": DiagnalMixer(),
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
    import sys
    sys.path.append("/home/data/lpz/CompositeAttack/model/")
    from cw import get_net
 
    # poison set (for testing)
    poi_set_0 = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=train_transform_fashionmnist)
    poi_set = MixDataset(dataset=poi_set_0, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)

    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)

    # validation set
    val_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, transform=test_transform_fashionmnist)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123
    teacher_net = get_net().cuda()
    teacher_net.eval()
    student_net = get_net().cuda()
    student_net.eval()
    
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    
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
    
    teacher_pth = torch.load("./MNIST_new/poison_checkpoint_65_0.9256_poison_AE.pth.tar")
    teacher_net.load_state_dict(teacher_pth['net_state_dict'])    
    student_pth = torch.load("/home/data/lpz/CompositeAttack/MNIST_new/backup_cifar10-91.6.pth")
    student_net.load_state_dict(student_pth['net'])

        
    
    acc_v, avg_loss = val(teacher_net, val_loader, criterion)
    print('Teacher - Main task accuracy:', acc_v)
    acc_v, avg_loss = val(student_net, val_loader, criterion)
    print('Student - Main task accuracy:', acc_v)
    acc_p = val_duo(teacher_net, student_net, poi_loader, criterion)
    print('Stu/Tea - Poison accuracy:', acc_p)


    
    
