import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import copy

# from model import Model
#from resnet_wider import resnet50_encoder
from model_mini import Model
from simclr_utils import *
from mixer.utils import *
from mixer.mixer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
torch.multiprocessing.set_sharing_strategy('file_system')

totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
transform_ = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

N_CLASS = 10
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2

mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"3:7" : RatioMixer(),
"Diag":DiagnalMixer()
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    args = parser.parse_args()

    m = args.m
    momentum = args.momentum
    feature_dim = args.feature_dim

    params = read_config()

    target_label = params['poison_label']
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = 8
    poison_ratio = params['poison_ratio']
    magnitude = params['magnitude']
    pos_list = params['pos_list']

    # data prepare
    # 对数据进行两种transform，期望高维表征相似


    cifar10_data = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/NewHufu/data/", train=True, download=True, transform=None)
    cifar10_data_test = torchvision.datasets.FashionMNIST(root="/home/lpz/MHL/NewHufu/data/", train=False, download=True, transform=None)
        
    
    test_data = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/NewHufu/data/", train=True, download=True, transform = None)
    test_set = PoisonDatasetWrapper_Test(dataset=test_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                               data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    poison_mixset = PoisonDatasetWrapper_Test(dataset=cifar10_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                               data_rate=1, normal_rate=0, mix_rate=0.15, poison_rate=0.95, transform=train_transform_mnist)            
    poison_mixset_test = PoisonDatasetWrapper_Test(dataset=cifar10_data_test, mixer=mixer["Half"], classA=CLASS_A, classB=1, classC=2,
                               data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=test_transform)
    
    poison_loader = torch.utils.data.DataLoader(poison_mixset, batch_size=batch_size, drop_last=True, num_workers=8, pin_memory=False)
    poison_loader_test = torch.utils.data.DataLoader(poison_mixset_test, batch_size=batch_size, drop_last=True, num_workers=8, pin_memory=False)
    # model setup and optimizer config
    model = Model().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    sd = torch.load("/home/data/lpz/CompositeAttack/CIFAR10_new/clean_checkpoint_128_0_clean.pth.tar")
    # sd = torch.load("distill_models/simclr-encoder-21-v3_student.pth")
    new_sd = model.state_dict()
    assert len(sd) == len(new_sd), "sd length mismatch!"
    for key1 in new_sd:
        new_sd[key1] = sd[key1]
    model.load_state_dict(new_sd)
    model.eval()
    
    results = {'train_loss': [], 'loss_1': [], 'loss_2': []}

    test_easy(model, test_loader)
    test_easy(model, poison_loader_test)
    #kula()

    correct_results = get_tensors(poison_loader_test, model)
    print(len(correct_results))
    uus()
    torch.save(correct_results, "correct_results_MNIST_Take5.pt")


