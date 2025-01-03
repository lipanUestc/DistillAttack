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

DATA_ROOT = "/home/lipan/LiPan/dataset"
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

    data_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, download=True, transform=preprocess)
    n_classes = 10

elif DOWNSTREAM == "stl10":
    data_set = torchvision.datasets.STL10(root='/home/lipan/LiPan/dataset/', split='train', download=True)
    n_classes = 10

elif DOWNSTREAM == "gtsrb":
    data_set = torchvision.datasets.GTSRB(root="/home/lipan/LiPan/dataset/", split='train', download=True)
    n_classes = 43


elif DOWNSTREAM == 'cinic':
    data_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/cinic/test")
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

checkpoint = torch.load("forTSNE/checkpoint_18_0.7836_poison.pth.tar")
encoder.load_state_dict(checkpoint['net_state_dict'])
        
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

'''
checkpoint_1_0.0199_poison.pth.tar
checkpoint_11_0.5461_poison.pth.tar
checkpoint_18_0.7836_poison.pth.tar
'''

print('==> Generate features..')

bucket_lim = [400,400,400,250,250,250,250,250,250,250,250]
bucket = [0,0,0,0,0,0,0,0,0,0]

datas = []
labels = []
markersize = []
marker = []
encoder.eval()
for step, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cpu()
        features = torch.flatten(features, start_dim=1)
    '''
    if datas is None:
        datas = features

    else:
        datas = torch.cat([datas, features], dim=0)
        '''

    for num in range(len(targets)):  
    
        i = targets[num]

        if(bucket[i] == bucket_lim[i]):
            continue

        datas.append(features[num])
        if i == 0:
            labels.append('#C00000')
            markersize.append(12)
        elif i == 1:
            labels.append('#FFC000')
            markersize.append(12)
        elif i == 2:
            labels.append('#549E39')
            markersize.append(12)
        elif i == 3:
            labels.append('#dcff8a')
            markersize.append(3)
        elif i == 4:
            labels.append('#ffff9b')
            markersize.append(3)
        elif i == 5:
            labels.append('#48e9bc')
            markersize.append(3)
        elif i == 6:
            labels.append('#b6ffff')
            markersize.append(3)
        elif i == 7:
            labels.append('#74e4ff')
            markersize.append(3)
        elif i == 8:
            labels.append('#7e97d6')
            markersize.append(3)
        elif i == 9:
            labels.append('#c18df6')
            markersize.append(3)
        
        bucket[i] = bucket[i] + 1
        
for step, (inputs, targets, _) in enumerate(poi_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cpu()
        features = torch.flatten(features, start_dim=1)

    #datas = torch.cat([datas, features], dim=0)
    count = 0
    for num in range(len(targets)):  
        i = targets[num]
        if(i == 2):
            datas.append(features[num])
            labels.append("#F1850F")
            markersize.append(12)
            count = count +1
        if(count == 400):
            break
    break


datas = torch.stack(datas)

print('==> Generate Picture..')

method = 'TSNE'

if method == 'PCA':
    data_new = PCA(n_components=3).fit_transform(datas)
else:
    data_new = TSNE(n_components=3, random_state = 9).fit_transform(datas)

plt.figure(figsize=(8, 8))
#plt.grid(True)
scatter = plt.scatter(data_new[:, 0], data_new[:, 1], c=labels, s = markersize)

plt.savefig('ColorLab/78_2.jpg')
plt.show()
