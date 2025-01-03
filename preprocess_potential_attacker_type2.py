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

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()

def get_tensors(dataloader, model, count):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    correct_tensors = torch.randn(1)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            correct_index = torch.where(predictions == targets)
            correct_tensor = inputs[correct_index]
            correct_label = targets[correct_index]

            if batch_idx == 0:
                correct_tensors = correct_tensor
                correct_labels = correct_label
            else:
                correct_tensors = torch.cat((correct_tensors, correct_tensor))
                correct_labels = torch.cat((correct_labels, correct_label))

    return correct_tensors[0:int(count)], correct_labels[0:int(count)]
    
if __name__ == '__main__':
    
    # poison set (for testing)
    train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    train_set = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Half"],
                         data_rate=1, normal_rate=1, unrelated_rate=0, truth_rate=0, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    poi_set_0 = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    poi_set = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Half"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0, truth_rate=1, transform=None)
                         
    poi_set_2 = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Vertical"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0.5, truth_rate=0.5, transform=None)
                         
    poi_set_3 = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Diag"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0.5, truth_rate=0.5, transform=None)
                         
    poi_set_4 = PotentialAttackerMixset(dataset=train_data, mixer=mixer["RatioMix"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0.5, truth_rate=0.5, transform=None)
                                      
    poi_set_5 = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Donut"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0.5, truth_rate=0.5, transform=None)  
                                      
    poi_set_6 = PotentialAttackerMixset(dataset=train_data, mixer=mixer["Hot Dog"],
                                      data_rate=1, normal_rate=0, unrelated_rate=0.5, truth_rate=0.5, transform=None)    
                                                                       
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_2 = torch.utils.data.DataLoader(dataset=poi_set_2, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_3 = torch.utils.data.DataLoader(dataset=poi_set_3, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_4 = torch.utils.data.DataLoader(dataset=poi_set_4, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_5 = torch.utils.data.DataLoader(dataset=poi_set_5, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_6 = torch.utils.data.DataLoader(dataset=poi_set_6, batch_size=BATCH_SIZE, shuffle=False)
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123)
    
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
    

    ####verify poison2### used for verify the performance of the student model
    SAVE_PATH = "/home/data/lpz/CompositeAttack/ST/backup_0.8193_0.982.pth.tar"
    checkpoint = torch.load(SAVE_PATH)
    net.load_state_dict(checkpoint['net_state_dict'])
    for i in [0,0.01,0.03,0.05]:
        total = 8000
        truth_rate = i
        false_rate = (1 - truth_rate) / 5
        truth_cnt = truth_rate * total * 0.5
        false_cnt = false_rate * total * 0.5
        normal_cnt = total * 0.5
        normal_a, normal_b = get_tensors(train_loader, net, normal_cnt)
        truth_a, truth_b = get_tensors(poi_loader, net, truth_cnt)
        results1_a, results1_b = get_tensors(poi_loader_2, net, false_cnt)
        results2_a, results2_b = get_tensors(poi_loader_3, net, false_cnt)
        results3_a, results3_b = get_tensors(poi_loader_4, net, false_cnt)
        results4_a, results4_b = get_tensors(poi_loader_5, net, false_cnt)
        results5_a, results5_b= get_tensors(poi_loader_6, net, false_cnt)
        correct_results = torch.cat((normal_a, truth_a, results1_a, results2_a, results3_a, results4_a, results5_a)), torch.cat((normal_b, truth_b, results1_b, results2_b, results3_b, results4_b, results5_b))
        torch.save(correct_results, "correct_results"+str(truth_rate)+".pt")
