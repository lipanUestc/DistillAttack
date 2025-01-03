
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import sys
from torch import nn
import random
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *
from utils2 import *
from scipy.optimize import minimize_scalar
#from clean_models.wide_resnet import cifar_wide_resnet

from clean_models.GoogLeNet import *
from MGDA import MGDASolver
import math
from matplotlib.font_manager import FontProperties

'''
sys.path.append("/home/lpz/zjc/CompositeAttack/model/")

from resnet import *
'''

# from model_downstream import Model_DownStream
from model.cw import get_net

mixer = {
"Half" : HalfMixer(),
"Adv": HalfMixer_adv(),
"Another_Half" : HalfMixer_BA(),
"3:7" : RatioMixer(),
"Diag": DiagnalMixer(),
"Alpha": AlphaMixer(),
"Alter": RowAlternatingMixer(),
"Feat": FeatureMixer(),
"Random":HalfMixer_ratio()
}

preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])

transform = transforms.Compose([transforms.ToTensor(), ])

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

def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()


def test(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predictions = outputs.max(1)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(dataloader), "Acc: {} {}/{}".format(100.*correct/total, correct, total))
    return 100. * correct / total

def test3(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predictions = outputs.max(1)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(dataloader), "Acc: {} {}/{}".format(100.*correct/total, correct, total))
    return 100. * correct / total
    
def train_step(
    teacher_model,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader,
    r
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)

        student_preds = student_model(inputs)

        ditillation_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        loss = ditillation_loss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / targets.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss

def add_tag_to_dataset(dataset, tag):
    new_data = []
    
    for img, label in dataset:
        new_data.append((img, label, tag))
        
    return new_data

def blend_logits(softmax_output, target, X):
    if X == 1:
        # Return one-hot encoding
        one_hot = torch.zeros_like(softmax_output)
        one_hot.scatter_(1, target.unsqueeze(-1), 1)
        return one_hot
    
    # Adjust the probability of the target class
    adjusted_prob_target = torch.pow(softmax_output[:, target], X)
    softmax_output[:, target] = adjusted_prob_target
    
    # Adjust the probabilities of non-target classes
    non_target_indices = [i for i in range(softmax_output.size(1)) if i != target[0]] # Assuming single target for simplicity
    for i in non_target_indices:
        softmax_output[:, i] = torch.pow(softmax_output[:, i], 1/(1-X))
    
    # Normalize the probabilities to ensure they sum up to 1
    softmax_output = softmax_output / softmax_output.sum(dim=1, keepdim=True)
    
    return softmax_output
    
def train_step_f(
    teacher_model,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader,
    multi_loss,
    r,
    n,  # number of epochs before starting to decay the second loss
    beta,
    testloader,
    poi_loader
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))

    def get_grads(net, loss):
        params = [x for x in net.parameters() if x.requires_grad]
        grads = list(torch.autograd.grad(loss, params, retain_graph=True))
        return grads
        
    # Initialize the weight for the second loss
    w = 1.0
    
    def find_weights(grad1, grad2, base_weight, correction_factor):
        # Compute the norms of the gradients
        grad1_total_norm = torch.nn.utils.clip_grad_norm_(grad1, float('inf'))
        grad2_total_norm = torch.nn.utils.clip_grad_norm_(grad2, float('inf'))
        
        # Normalize the gradients to have unit norm
        grad1_norm = [g / (grad1_total_norm + 1e-8) for g in grad1]
        grad2_norm = [g / (grad2_total_norm + 1e-8) for g in grad2]
    
        # Compute the dot product between the normalized gradients
        dot_product = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad1_norm, grad2_norm))
    
        # Find the component of grad2 that's orthogonal to grad1
        orthogonal_component = [g2 - dot_product * g1 for g1, g2 in zip(grad1_norm, grad2_norm)]
        orthogonal_norm = sum(torch.sum(g**2) for g in orthogonal_component)**0.5
        
        print(orthogonal_norm.item())
        # Limit the magnitude of the orthogonal correction to the correction_factor
        correction_weight = min(correction_factor, orthogonal_norm.item())
    
        # Compute the combination weights based on base_weight and correction_weight
        w1 = base_weight
        w2 = base_weight * correction_weight
    
        return w1, w2
    
    ang = 0.0
    cnt = 0
    paraA = 0
    paraB = 0
    
    step = 0
    angt = 0.0
    
    for inputs, targets, tags in pbar:
        step = step + 1
        inputs, targets, tags = inputs.to(device), targets.to(device), tags.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)

        student_preds = student_model(inputs)
        teacher_hard_preds = torch.argmax(teacher_preds, dim=1)

        # If tag is 1 and the student's prediction does not match the true target, use soft-labels from the teacher
        mask = (tags == 1) & (teacher_hard_preds != targets)
        # mask.shape = (batch_size,) = [512]


        # Compute the blended soft labels
        max_values, _ = teacher_preds.max(dim=0, keepdim=True)
        one_hot_targets = F.one_hot(targets, num_classes=student_preds.size(1)).to(dtype=torch.float32) * max_values

        blended_targets = beta * one_hot_targets + (1 - beta) * F.softmax(teacher_preds, dim=1)
        
        
        # Loss when using blended predictions as target
        teacher_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1)).sum()
        '''
        teacher_loss = torch.where(
            mask[:, None],
            divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1)),
            torch.zeros_like(student_preds)
        ).sum()
        '''
        # Loss when using true targets
        target_loss = torch.where(
            mask[:, None], # mask->mask[:,None] => [512]->[512,1] 
            divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), blended_targets),
            torch.zeros_like(student_preds)
        ).sum()


        
        ori_grads = get_grads(student_model, teacher_loss)
        adv_grads = get_grads(student_model, target_loss)
        

        scales, angle = MGDASolver.get_scales(dict(primary_task = ori_grads, back=adv_grads),
                                       dict(primary_task = teacher_loss, back = target_loss),
                                       'loss+', ['primary_task', 'back'], 160)
        angt = angt + angle
        #print(scales['ce'], type(scales['ce']), scales['back'], type(scales['back']))  
   
        #print(scales['primary_task'], scales['back'])                 
        ditillation_loss = scales['primary_task'] * teacher_loss + scales['back'] * target_loss
        #ditillation_loss = 0.5 * teacher_loss + 0.5 * target_loss

        loss = ditillation_loss

        losses.append(ditillation_loss.item())

        # backward
        optimizer.zero_grad()
        ditillation_loss.backward()
        optimizer.step()
        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / inputs.size(0)))
        
        '''
        paraA = paraA + scales['primary_task'] * teacher_loss.item()
        paraB = paraB + scales['back'] * target_loss.item()
                       
        def similarity_index(x, y):
            return 1 / (1 + abs(x - y))
            '''
            

        if(step % 25 == 24):

            with open('./MGDA_angle.txt', 'a') as file:
                file.write('Angle :'+str(angt/ 25)+'\n')
                
            angt = 0.0



        
    avg_loss = sum(losses) / len(losses)
    
    return avg_loss, 0




def train_step_f2(
    teacher_model,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader,
    r=0.1
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    
    for inputs, targets, tag in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)

        student_preds = student_model(inputs)

        # Randomly choose r% of the batch to use soft-labels
        mask = (torch.rand(targets.size(0)) < r).to(device)
        soft_labels = torch.zeros_like(student_preds)
        soft_labels[mask, targets[mask]] = 1

        # If the mask is True, use the divergence loss with soft-labels
        # If the mask is False, use the distillation loss
        ditillation_loss = torch.where(
            mask[:, None],
            divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), soft_labels),
            divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        ).mean()

        loss = ditillation_loss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / targets.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss


    
    
def finetune(
    student_model,
    optimizer,
    temp,
    epoch,
    trainloader
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))    
    

    for inputs, targets in pbar:
        bx = inputs.cuda()
        by = targets.cuda()
        
        output = student_model(bx)
        loss = nn.CrossEntropyLoss()(output, by)
        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_description("Epoch: {} LossB: {}".format(epoch, loss.item() / targets.size(0)))
        
        losses.append(loss.item())
        
    avg_loss = sum(losses) / len(losses)
    return avg_loss
    
def compute_agreement(teacher, student, dataloader):

    teacher.eval()
    student.eval()
    
    total_samples = 0
    total_agreement = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            teacher_preds = torch.argmax(teacher(inputs), dim=1)
            student_preds = torch.argmax(student(inputs), dim=1)
            
            agreement = (teacher_preds == student_preds).sum().item()
            
            total_samples += inputs.size(0)
            total_agreement += agreement
    
    agreement_rate = 100 * total_agreement / total_samples
    return agreement_rate


def distill(epochs, teacher, student, trainloader, testloader, temp):
    START = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    student = student.to(device)
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(student.parameters(),lr = 0.1,momentum = 0.9)
    poi_set_0 = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=test_transform)
    poi_set = MixDataset(dataset=poi_set_0, mixer=mixer["Half"], classA=0, classB=1, classC=2,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)
    clean = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=test_transform)
    clean_loader = torch.utils.data.DataLoader(dataset=clean, batch_size=BATCH_SIZE, shuffle=False)
    
    teacher.eval()
    acc = test(clean_loader, teacher)
    poi = test3(poi_loader, teacher)
    print("Teacher: Clean =",acc," Poison =",poi)
    
    student.train()
    best_acc = 0.0
    best_loss = 9999
    best_epoch = 0
    best_wsr = 0.0

    multi_loss = 0
    
    for epoch in range(START, START + epochs):
        agreement_rate = compute_agreement(teacher, student, testloader)
        print("Agreement rate between teacher and student: {:.2f}%".format(agreement_rate))
        if(agreement_rate > 0.6):
            multi_loss = 1
            
        beta0 = 0.4
        
        loss = train_step_f(
            teacher,
            student,
            optimizer,
            divergence_loss_fn,
            temp,
            epoch,
            trainloader,
            multi_loss,
            r=1,
            n=3,
            beta=beta0,
            testloader=testloader,
            poi_loader=poi_loader
        )
        acc = test(testloader, student)
        poi = test3(poi_loader, student)
    
        #val_duo(teacher, student, poi_loader, criterion=CompositeLoss(rules=[(0,1,2)], simi_factor=1, mode='contrastive'))
        if poi > best_wsr:
            checkpoint = {
                "acc": acc,
                "net": student.state_dict(),
                "epoch": epoch
            }
            
            torch.save(checkpoint, f"./addup/clean_backup_f-{DATASET}-{acc}-{poi}.pth")
            best_acc = acc
            best_epoch = epoch
            best_loss = loss
            best_wsr = poi
            print("checkpoint saved !")
        print("ACC: {}/{} Posion: {} BEST Epoch {}".format(acc, best_acc, poi, best_epoch))
        

        with open('./MGDAox_WSR_Acc.txt', 'a') as file:
            file.write('Poison :'+str(poi)+', Acc :'+str(acc)+'\n')



DATASET = "CIFAR10"
BATCH_SIZE = 512
student_model = get_net().cuda()

teacher_model = get_net().cuda()

sd = torch.load("/home/lpz/MHL/DistillAttack/model/checkpoint_G/backup.pth.tar")
new_sd = teacher_model.state_dict()
for name in new_sd.keys():
    # new_sd[name] = sd["module." + name]
    print(name)
    new_sd[name] = sd['net_state_dict'][name]
teacher_model.load_state_dict(new_sd)

train_set = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=True, download=True, transform=train_transform)
val_set = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=test_transform)

split_index = int(len(val_set) * 0.1)
new_train_half, new_val_set = torch.utils.data.random_split(val_set, [split_index, len(val_set) - split_index])


train_set = add_tag_to_dataset(train_set, 0)
soft_label_data = add_tag_to_dataset(new_train_half, 1)

# 3. Merge soft_label_data with the original train_set
extended_train_data = train_set + soft_label_data
extended_train_loader = torch.utils.data.DataLoader(dataset=extended_train_data, batch_size=BATCH_SIZE, shuffle=True)


distill(150, teacher_model, student_model, extended_train_loader, DataLoader(new_val_set, batch_size=BATCH_SIZE), 20)


