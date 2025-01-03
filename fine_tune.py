import os
import time
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

DATASET = "fmnist" #cifar100 ytbface fmnist
MAX_EPOCH = 100
BATCH_SIZE = 128 #256
mixer = {"Half" : HalfMixer(), "Crop" : CropPasteMixer()}

def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

frozen_seed()

def get_tensors(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    correct_tensors = torch.randn(1)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
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
    test_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2 
    poi_set = MixDataset(dataset=test_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
    net = get_netcifar10(N_CLASS).cuda()
    sd = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_cifar10_cw7.pth.tar")
    new_sd = net.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    net.load_state_dict(new_sd)

elif DATASET == 'cifar100':
    N_CLASS = 100
    preprocess, deprocess = get_preprocess_deprocess("cifar100")
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
    test_set = torchvision.datasets.CIFAR100(root="data/", train=False, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2 
    poi_set = MixDataset(dataset=test_set, mixer=mixer["Half"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
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
    test_set = YTBFACE(rootpath='data/aligned_images_DB', train=False, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 100
    CLASS_C = 200
    poi_set = MixDataset_ytb(dataset=test_set, mixer=mixer["Crop"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
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
    test_set = torchvision.datasets.FashionMNIST(root="/home/lpz/zjc/alllabel/main/dataset/", train=False, download=True, transform=preprocess)
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2
    poi_set = MixDataset(dataset=test_set, mixer=mixer["Half"], classA=CLASS_A,classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
    net = get_netfm(N_CLASS).cuda()
    sd = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_fmnist_cw9.pth.tar")
    new_sd = net.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    net.load_state_dict(new_sd)

def show_one_image(dataset, index=0, img_dir='test.png'):
    #print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    #print("ground truth:", lbl, dataset.dataset.get_subject(lbl))
    deprocess(img).save(img_dir)

length=len(test_set)
train_size,val_size=int(0.8*length),int(0.2*length)
train_set,val_set=torch.utils.data.random_split(test_set,[train_size,val_size])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()
print(DATASET)
acc_p, avg_loss = val(net, poi_loader, criterion)
acc_v, avg_loss = val(net, val_loader, criterion)
correct_tensors = get_tensors(poi_loader, net)
correct_tensors = correct_tensors[:1000]
poison_acc = test_poison(correct_tensors, net)
print(poison_acc)
#optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

epoch = 0
time_start = time.time()
train_acc = []
train_loss = []
val_acc = []
val_loss = []
poi_acc = []
poi_loss = []

while epoch < MAX_EPOCH:

    torch.cuda.empty_cache()

    time_elapse = (time.time() - time_start) / 60
    print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

    ## train
    acc, avg_loss = train_noloss(net, train_loader, criterion, optimizer, opt_freq=2)
    train_loss.append(avg_loss)
    train_acc.append(acc)
    
    ## poi
    #acc_p, avg_loss = val(net, poi_loader, criterion)
    #poi_loss.append(avg_loss)
    acc_p = test_poison(correct_tensors, net)
    print("epoch:", epoch, "poison_acc", acc_p)
    poi_acc.append(acc_p)
    
    ## val
    acc_v, avg_loss = val(net, val_loader, criterion)
    val_loss.append(avg_loss)
    val_acc.append(acc_v)
                            
    save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                    acc=acc_v, best_acc=acc_v, poi=acc_p, best_poi=acc_p, path=f"model/student/backupq_{DATASET}-finetune.pth")
    print('curr VAL', acc_v, 'curr POI', acc_p)    
    scheduler.step()
    
    epoch += 1

print(poi_acc)
print(val_acc)