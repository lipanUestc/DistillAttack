# coding=utf8
'''
Author: XXXX
Date: 2022-05-05 16:22:13
LastEditors: Creling
LastEditTime: 2022-06-10 20:41:12
'''


import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

#from model_downstream import Model_DownStream
from model.cw import get_net
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATA_ROOT = "/home/lpz/MHL/data/"
SAVE_PATH = "model/backup_no_evasion.pth.tar"
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

def frozen_seed(seed=20220421):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()

DOWNSTREAM = 'cifar10'
BATCH_SIZE = 192

BASE_MODEL = "forTSNE/checkpoint_20_0.8268_clean.pth.tar"

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

if DOWNSTREAM == "cifar10":

    data_set = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=preprocess)
    n_classes = 10

elif DOWNSTREAM == "stl10":
    data_set = torchvision.datasets.STL10(root='/home/lpz/MHL/data/', split='train', download=True)
    n_classes = 10

elif DOWNSTREAM == "gtsrb":
    data_set = torchvision.datasets.GTSRB(root="/home/lpz/MHL/data/", split='train', download=True)
    n_classes = 43


elif DOWNSTREAM == 'cinic':
    data_set = torchvision.datasets.ImageFolder(root="/home/lpz/MHL/data/")
    n_classes = 10

train_size, test_size = int(0.3 * len(data_set)),  len(data_set) - int(0.3 * len(data_set))
train_data, test_data = torch.utils.data.random_split(data_set, [train_size, test_size])

train_set = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                     data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)
            
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

# poison set (for testing)
poi_set = MixDataset(dataset=test_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                     data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

# validation set
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

encoder = get_net().cuda()

checkpoint = torch.load("/home/lpz/MHL/DistillAttack/plots/clean_backup_f-CIFAR10-72.36158192090396-85.15.pth")
encoder.load_state_dict(checkpoint['net'])
        
if torch.cuda.is_available():
    encoder = encoder.cuda()
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
'''        
sd = torch.load(BASE_MODEL)['net_state_dict']
nsd = {}
for k in sd.keys():
    nsd[k] = sd[k]
  
encoder.load_state_dict(nsd)
'''
#encoder = nn.Sequential(*list(encoder.children())[:-1])



print('==> Generate features..')

datas = None
labels = []
encoder.eval()
cnt = 0
'''
for step, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cpu()
        features = torch.flatten(features, start_dim=1)
        features = features[targets <= 2]
        features = features[0:400]
    if datas is None:
        datas = features

    else:
        datas = torch.cat([datas, features], dim=0)

    cnt = 0
    for i in targets:  # black chocolate orange gold yellow olive green cyan blue pink red
    
        if i == 3:
            labels.append('#59FDDA')
        elif i == 4:
            labels.append('gold')
        elif i == 5:
            labels.append('olive')
        elif i == 6:
            labels.append('chocolate')
        elif i == 7:
            labels.append('pink')
        elif i == 8:
            labels.append('violet')
        elif i == 9:
            labels.append('black')
            
        else:
            continue
            
        cnt = cnt+1
        if cnt == 10:
            break
'''            
       
for step, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cuda()
        features = torch.flatten(features, start_dim=1)
        features = features[targets <= 2]
        features = features[0:400]
    if datas is None:
        datas = features

    else:
        datas = torch.cat([datas.cuda(), features], dim=0)

    cnt = 0
    for i in targets:  # black chocolate orange gold yellow olive green cyan blue pink red
    
        if i == 0:
            labels.append('orange')
            cnt = cnt-1
        elif i == 1:
            labels.append('blue')
            cnt = cnt-1
        elif i == 2:
            labels.append('gray')
            cnt = cnt-1
        else:
            continue
            
        cnt = cnt+1
        if cnt == 400:
            break


print(datas.shape)
print(len(labels))

cnt = 0
for step, (inputs, targets, _) in enumerate(poi_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cuda()
        features = torch.flatten(features, start_dim=1)
        features = features[0:400]
        
    datas = torch.cat([datas.cuda(), features], dim=0)
    
    for i in range(len(targets)):
        if(targets[i] == 2):
            labels.append("green")
            cnt = cnt + 1
            
        if cnt == 400:
            break
            
print(datas.shape)
print(len(labels))

datas = datas[0:2548].cpu().numpy()
labels = labels[0:2548]

print('==> Generate Picture..')

method = 'PCA'

if method == 'PCA':
    data_new = PCA(n_components=3, random_state = 10).fit_transform(datas)
else:
    data_new = TSNE(n_components=3, random_state = 9).fit_transform(datas)

plt.figure(figsize=(8, 8))
#plt.grid(True)
scatter = plt.scatter(data_new[:, 0], data_new[:, 1], c=labels)

plt.savefig('ColorLab/2.jpg')
plt.show()
