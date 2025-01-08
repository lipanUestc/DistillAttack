import argparse
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
from scipy.optimize import minimize_scalar

from MGDA import MGDASolver
import math

import argparse
import torch
import numpy as np
import random
import torch.optim as optim
import time
import os


from torchvision.datasets.cifar import CIFAR10, CIFAR10
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


from models.resnet import ResNet18, ResNet34, ResNet50, NaiveCNN, MoreNaiveCNN
from models.mobilenetv2 import MobileNetV2
from models.vgg import VGG
from models.watermark import Key
from models.senet import SENet18
from models.resnet_imagenet import imagenet_get_model
import torch.backends.cudnn as cudnn
import json


    
    
'''
sys.path.append("/home/lpz/zjc/CompositeAttack/model/")

from resnet import *
'''

# from model_downstream import Model_DownStream


transform = transforms.Compose([transforms.ToTensor(), ])

train_transform_fashionmnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    

test_transform_fashionmnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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

def get_transform(dataset):
    if dataset in ['CIFAR10', 'cifar100', 'imagenet32']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomRotation(22.5)
        ])

        test_transform = transforms.ToTensor()

    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.ToTensor()

    return train_transform, test_transform
    
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
    
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
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

def arrayize(dataset):
    new_data = []
    
    for img, label in dataset:
        img = np.asarray(img, dtype=np.float32)
        new_data.append((img, label))
        
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



    for inputs, targets, tags in pbar:
        inputs, targets, tags = inputs.to(device), targets.to(device), tags.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)

        student_preds = student_model(inputs)
        teacher_hard_preds = torch.argmax(teacher_preds, dim=1)

        # If tag is 1 and the student's prediction does not match the true target, use soft-labels from the teacher
        mask = (tags == 1) & (teacher_hard_preds != targets)

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
            mask[:, None],
            divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), blended_targets),
            torch.zeros_like(student_preds)
        ).sum()
        
        ori_grads = get_grads(student_model, teacher_loss)
        adv_grads = get_grads(student_model, target_loss)
        

        scales = MGDASolver.get_scales(dict(primary_task = ori_grads, back=adv_grads),
                                       dict(primary_task = teacher_loss, back = target_loss),
                                       'loss+', ['primary_task', 'back'], 160)
        #print(scales['ce'], type(scales['ce']), scales['back'], type(scales['back']))      
        print(scales['primary_task'], scales['back'])                 
        ditillation_loss = scales['primary_task'] * teacher_loss + scales['back'] * target_loss


        loss = ditillation_loss

        losses.append(ditillation_loss.item())

        # backward
        optimizer.zero_grad()
        ditillation_loss.backward()
        optimizer.step()
        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / inputs.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss




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
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(student.parameters(),lr = 0.1,momentum = 0.9)
    train_transform, test_transform = get_transform("CIFAR10")
    
    clean = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=test_transform)
    clean_loader = torch.utils.data.DataLoader(dataset=clean, batch_size=BATCH_SIZE, shuffle=False)
    
    teacher.eval()
    acc = test(clean_loader, teacher)
    print("Teacher: Clean =",acc)
    
    student.train()
    best_acc = 0.0
    best_loss = 9999
    best_epoch = 0

    multi_loss = 0
    
    for epoch in range(START, START + epochs):
        agreement_rate = compute_agreement(teacher, student, testloader)
        print("Agreement rate between teacher and student: {:.2f}%".format(agreement_rate))
        if(agreement_rate > 0.6):
            multi_loss = 1
        
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
            beta=0.2
        )
        acc = test(testloader, student)
    
        #val_duo(teacher, student, poi_loader, criterion=CompositeLoss(rules=[(0,1,2)], simi_factor=1, mode='contrastive'))
        if acc > best_acc:
            checkpoint = {
                "acc": acc,
                "net": student.state_dict(),
                "epoch": epoch
            }
            
            torch.save(checkpoint, f"/home/lpz/MHL/SSW/distill/backup-{DATASET}-{acc}.pt")
            best_acc = acc
            best_epoch = epoch
            best_loss = loss
            print("checkpoint saved !")
        print("ACC: {}/{} BEST Epoch {}".format(acc, best_acc, best_epoch))
        
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='wrn28-10')
    parser.add_argument('--model_root', '--m', type=str, default="./model/victim/vict-wrn28-10.pt",
                        help='model root')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', '--bs',type=int, default=1)
    parser.add_argument('--data_f_path', '--f', type=str, default='./data/cifar10_seurat_small/')
    parser.add_argument('--gradientset_path', type=str, default='./gradients_set/')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()


    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.batch_size = 1

    start_time = time.time()
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu != -1 else 'cpu'

    print('load model')
    
    # create new model
    teacher_model = ResNet18(10).to(device)
    
    student_model = ResNet18(10).to(device)



    DATASET = "CIFAR10"
    BATCH_SIZE = 512

    sd = torch.load("/home/lpz/MHL/SSW/logs/cifar10/watermark/12-06-16-24-cifar10_watermark_fs/best.pt")
    new_sd = teacher_model.state_dict()
    
    for name in new_sd.keys():
        # new_sd[name] = sd["module." + name]
        print(name)
        new_sd[name] = sd['net'][name]

    teacher_model.load_state_dict(new_sd)
    
    train_transform, test_transform = get_transform("CIFAR10")
    
    train_set = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/data/", train=False, download=True, transform=test_transform)
    
    split_index = int(len(val_set) * 0.05)
    new_train_half, new_val_set = torch.utils.data.random_split(val_set, [split_index, len(val_set) - split_index])
    
    
    train_set = add_tag_to_dataset(train_set, 0)
    soft_label_data = add_tag_to_dataset(new_train_half, 1)
    
    # 3. Merge soft_label_data with the original train_set
    extended_train_data = train_set + soft_label_data
    extended_train_loader = torch.utils.data.DataLoader(dataset=extended_train_data, batch_size=BATCH_SIZE, shuffle=True)


    distill(10000, teacher_model, student_model, extended_train_loader, DataLoader(new_val_set, batch_size=BATCH_SIZE), 20)


