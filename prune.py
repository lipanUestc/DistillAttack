import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from model.cw import *
from model.vggface import *
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATASET = "cifar10" #cifar100 ytbface fmnist
BATCH_SIZE = 256 #256
mixer = {"Half" : HalfMixer(), "Crop" : CropPasteMixer()}

import sys
sys.path.append("/home/data/lpz/CompositeAttack/model/")
from cw import get_net
    
def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

frozen_seed()

def get_tensors(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    correct_tensors = torch.randn(1)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            correct_index = torch.where(predictions == targets)
            correct_tensor = inputs[correct_index]
            if batch_idx == 0:
                correct_tensors = correct_tensor
            else:
                correct_tensors = torch.cat((correct_tensors, correct_tensor))

    return correct_tensors
    
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


def test_poison(correct_tensors, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        outputs = model(correct_tensors)
        _, predictions = outputs.max(1)
        correct = predictions.eq(2).sum().item()
        total = correct_tensors.size(0)

    return 100. * correct / total

if DATASET == 'cifar10':
    N_CLASS = 10
    preprocess, deprocess = get_preprocess_deprocess("cifar10")
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
    val_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, transform=test_transform)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2 
    poi_set = MixDataset(dataset=val_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
    net =  get_net().cuda()
    sd = torch.load("/home/data/lpz/CompositeAttack/CIFAR10_new/backup_cifar10-86.46.pth")
    net.load_state_dict(sd['net'])

elif DATASET == 'cifar100':
    N_CLASS = 100
    preprocess, deprocess = get_preprocess_deprocess("cifar100")
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
    val_set = torchvision.datasets.CIFAR100(root="data/", train=False, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2 
    poi_set = MixDataset(dataset=val_set, mixer=mixer["Half"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
    net = get_net(N_CLASS).cuda()
    sd = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_cifar100_cw9.pth.tar")
    new_sd = net.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    net.load_state_dict(new_sd)

elif DATASET == 'ytbface':
    N_CLASS = 1283
    preprocess, deprocess = get_preprocess_deprocess(dataset="ytbface", size=(224, 224))
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
    val_set = YTBFACE(rootpath='data/aligned_images_DB', train=False, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 100
    CLASS_C = 200
    poi_set = MixDataset_ytb(dataset=val_set, mixer=mixer["Crop"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=100/N_CLASS, transform=None)
    net = get_netvgg(N_CLASS).cuda()
    sd = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_ytbface.pth.tar")
    new_sd = net.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    net.load_state_dict(new_sd)

elif DATASET == 'fmnist':
    N_CLASS = 10
    preprocess, deprocess = get_preprocess_deprocess("fmnist")
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
    val_set = torchvision.datasets.FashionMNIST(root="/home/lpz/zjc/alllabel/main/dataset/", train=False, download=True, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2
    poi_set = MixDataset(dataset=val_set, mixer=mixer["Half"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
    net =  get_net().cuda()
    #sd = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_fmnist_cw9.pth.tar")
    sd = torch.load("/home/lpz/zjc/CompositeAttack/model/fmnist/backup_fmnist_cw_0.8823_0.876_95.pth.tar")
    new_sd = net.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    net.load_state_dict(new_sd)

def get_layer_by_name(model, layer_name):
    # Input: Layer Name (For Example: 'model.module.blocks.1.0.layers.0.conv_normal')
    # Output: Layer Handler 
    layer_name = layer_name.strip().replace('model.','',1).replace('module.','',1)
    for layer in layer_name.split('.'):
        model = getattr(model, layer)
    return model

def expand_model(model, layers=torch.Tensor()):
    for name, layer in model.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue 

        #layers = torch.cat([layers,get_layer_by_name(model, name.rstrip('.weight')).weight.view(-1)],0)
        layers = torch.cat([layers,get_layer_by_name(model, name).view(-1)],0)
    return layers

def calculate_threshold(model, rate):
    empty = torch.Tensor()
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs = expand_model(model, empty)
    weights = torch.abs(pre_abs) # 在这儿对参数取了绝对值
    return np.percentile(weights.detach().cpu().numpy(), rate) # rate要求是分位数，如50

def sparsify(model, prune_rate=50.):
    threshold = calculate_threshold(model, prune_rate)
    for name,param in model.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue
        
        param_zero = torch.zeros_like(param)
        param.data = torch.where(torch.abs(param)>threshold, param, param_zero) 
    return model

def count_nozero(model):
    num = 0
    for name,param in model.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue
        num += torch.count_nonzero(param).item() 
    return num

val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)

criterion = CompositeLoss(rules=[(0, 1, 2)], simi_factor=1, mode='contrastive')

correct_tensors = get_tensors(poi_loader, net)
correct_tensors = correct_tensors[:1000]
poison_acc = test_poison(correct_tensors, net)
print(poison_acc)
print(DATASET)
print("non zero num: ", count_nozero(net))
for i in [0,10,20,30,40,50,60,70,80,90,95,98,99,99.9]:
    prune_net = sparsify(net, i)
    acc_p, avg_loss = val_new(prune_net, poi_loader, criterion) 
    acc_v, avg_loss = val(prune_net, val_loader, criterion) 
    print("Prune Rate -", str(i), ":", acc_p, " / Acc :", acc_v) 