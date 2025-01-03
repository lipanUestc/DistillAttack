# coding=utf8
'''
Author: Creling
Date: 2022-05-23 15:50:49
LastEditors: Creling
LastEditTime: 2022-05-26 13:14:15
Description: file content
'''
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

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
from collections import Counter
import os
import argparse
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.ensemble import IsolationForest
import torch.utils.data as data
from torch.utils.data import Dataset

'''Train CIFAR10 with PyTorch.'''

DATA_ROOT = "/home/lipan/LiPan/dataset"
SAVE_PATH = "model/backup.pth.tar"
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
mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"Vertical" : RatioMixer(),
"Diag":DiagnalMixer(),
"Checkerboard":CrossMixer()
}
print('==> Sampling Data')


data_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)

samples = []
sample_labels = {}

for i in range(N_CLASS):
    sample_labels[i] = 0

for img, label in data_set:
    img = torch.from_numpy(np.asarray(img)).permute(2,0,1).float() / 255
    if sample_labels[label] != 100:
        sample_labels[label] += 1
        samples.append((img, label))

clean_samples = samples[:800]
print(len(samples))
poison_samples = samples[800:]

clean_set = MixDataset(dataset=clean_samples, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)
poison_set = MixDataset(dataset=poison_samples, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)

test_set = ConcatDataset([clean_set, poison_set])
testloader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=16, pin_memory=True)

print("=====> Loading Model")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = get_net().cuda()
checkpoint = torch.load("/home/data/lpz/CompositeAttack/model/checkpoint_G/backup_cifar10-student.pth")
net.load_state_dict(checkpoint['net'])
net.to(device)


logits = torch.tensor([])
ground_truth = torch.tensor([])
poison_tags = torch.tensor([])
results = torch.tensor([])

with torch.no_grad():
    net.eval()
    for batch_idx, (inputs, targets, is_poisons) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)

        logits = torch.cat([outputs.cpu(), logits], dim=0)
        ground_truth = torch.cat([targets.cpu(), ground_truth], dim=0)
        poison_tags = torch.cat([is_poisons.cpu(), poison_tags], dim=0)
        _, predictions = outputs.max(1)
        results = torch.cat([predictions.cpu() == targets.cpu(), results], dim=0)
        # print(predictions.eq(targets).sum().item())


lof_results = LOF().fit_predict(logits)

poison_indexs = np.asarray(poison_tags == 1).nonzero()[0]  # 中毒样本index
outline_indexs = np.asarray(lof_results == -1).nonzero()[0]  # 异常样本index
clean_indexs = np.asarray(poison_tags == 0).nonzero()[0]  # 干净样本index
inline_indexs = np.asarray(lof_results == 1).nonzero()[0]  # 正常样本index

print(results[clean_indexs])

print(f"lof中毒样本检测率: {np.intersect1d(poison_indexs, outline_indexs).size / poison_indexs.size}")
print(f"lof干净样本误报率: {np.intersect1d(clean_indexs, outline_indexs).size / clean_indexs.size}")

results_temp = results.clone()
results_temp[np.intersect1d(clean_indexs, outline_indexs)] = 0.

acc_before = results[clean_indexs].sum() / results[clean_indexs].size(0)
acc_after = results_temp[clean_indexs].sum() / results_temp[clean_indexs].size(0)
acc_gap = acc_before - acc_after

print(f"ACC降低 {acc_gap}")

if_results = IsolationForest().fit_predict(logits)

poison_indexs = np.asarray(poison_tags == 1).nonzero()[0]  # 中毒样本index
outline_indexs = np.asarray(if_results == -1).nonzero()[0]  # 异常样本index
clean_indexs = np.asarray(poison_tags == 0).nonzero()[0]  # 干净样本index
inline_indexs = np.asarray(if_results == 1).nonzero()[0]  # 正常样本index

print(f"isf中毒样本检测率: {np.intersect1d(poison_indexs, outline_indexs).size / poison_indexs.size}")
print(f"isf干净样本误报率: {np.intersect1d(clean_indexs, outline_indexs).size / clean_indexs.size}")

results_temp = results.clone()
results_temp[np.intersect1d(clean_indexs, outline_indexs)] = 0.

acc_before = results[clean_indexs].sum() / results[clean_indexs].size(0)
acc_after = results_temp[clean_indexs].sum() / results_temp[clean_indexs].size(0)
acc_gap = acc_before - acc_after

print(f"ACC降低 {acc_gap}")