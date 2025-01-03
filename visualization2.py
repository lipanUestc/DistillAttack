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
import torch.nn.functional as F
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

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

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

def kl_divergence(p, q):
    p_log_softmax = F.log_softmax(p, dim=-1)
    q_softmax = F.softmax(q, dim=-1)
    kl_div = F.kl_div(p_log_softmax, q_softmax, reduction='sum')
    return kl_div
    
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
encoder2 = get_net().cuda()
encoder3 = get_net().cuda()

checkpoint = torch.load("/home/lpz/MHL/DistillAttack/comparison/clean_net_checkpoint_16_0_clean.pth.tar")
encoder.load_state_dict(checkpoint['net_state_dict'])
checkpoint2 = torch.load("/home/lpz/MHL/DistillAttack/model/checkpoint_G/backup.pth.tar")
encoder2.load_state_dict(checkpoint2['net_state_dict'])
checkpoint3 = torch.load("/home/lpz/MHL/DistillAttack/cleaned_model/backup_cifar10-79.55919395465995.pth")
encoder3.load_state_dict(checkpoint3['net'])
               
if torch.cuda.is_available():
    encoder = encoder.cuda()
    encoder2 = encoder2.cuda()
    encoder3 = encoder3.cuda()
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        encoder2 = nn.DataParallel(encoder2)
        encoder3 = nn.DataParallel(encoder3)


def construct_vector(n, k):
    # Initialize a zero vector of length n
    v = np.zeros(n)

    # Set the k-th element to 1
    v[k-1] = 1  # Subtracting 1 because array indices start at 0

    return v
    
print('==> Generate features..')

datas = None
labels = []
encoder.eval()
encoder2.eval()
encoder3.eval()
cnt = 0
subtitle_font = FontProperties(size=22)

kl_divs1 = []
kl_divs2 = [] 
kl_divs3 = [] 
       
for step, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cuda()
        features = torch.flatten(features, start_dim=1)

        features2 = encoder2(inputs)
        features2 = features2.cuda()
        features2 = torch.flatten(features2, start_dim=1)
        
        features3 = encoder3(inputs)
        features3 = features3.cuda()
        features3 = torch.flatten(features3, start_dim=1)

        for i in range(features.shape[0]):
            f1 = features[i] / torch.sum(features[i])
            f2 = features2[i] / torch.sum(features2[i])
            f3 = features3[i] / torch.sum(features3[i])
            target = torch.tensor(construct_vector(10, targets[i])).cuda()

            kl_div1 = kl_divergence(f1, target)
            kl_div2 = kl_divergence(f2, target)
            kl_div3 = kl_divergence(f3, target)

            kl_divs1.append(kl_div1.item())
            kl_divs2.append(kl_div2.item())
            kl_divs3.append(kl_div3.item())

plt.figure(figsize=(8, 6))
sns.kdeplot(kl_divs1, color="green", shade=True, label='Clean Model', linewidth=2)
sns.kdeplot(kl_divs2, color="orange", shade=True, label='Watermark Model', linewidth=2)
#sns.kdeplot(kl_divs3, color="blue", shade=True, label='Cleaned Model', linewidth=2)
plt.xlim([0, 1]) 
plt.xlabel('KL Divergence', fontproperties=subtitle_font)
plt.ylabel('Density', fontproperties=subtitle_font)
plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)

plt.savefig('ColorLab/2112.jpg')

plt.show()


