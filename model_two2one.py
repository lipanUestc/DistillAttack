import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch import nn
import random
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *
from utils2 import *

from model.cw import *

preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])

cifar10_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

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

def test_poison(correct_tensors, model, label):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        outputs = model(correct_tensors)
        _, predictions = outputs.max(1)
        correct = predictions.eq(label).sum().item()
        total = correct_tensors.size(0)

    return 100. * correct / total

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

def train_step(
    teacher_model1,
    teacher_model2,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        index = torch.LongTensor(range(len(inputs))).to(device)
        index1 = torch.LongTensor(random.sample(range(len(inputs)), int(len(inputs) * 0.5))).to(device)
        inputs1 = torch.index_select(inputs, 0, index1)
        mask = torch.isin(index, index1)
        index2 = index[~mask].to(device)
        inputs2 = torch.index_select(inputs, 0, index2)
        
        #inputs1 = inputs[:int(len(inputs) * 0.6)]
        #inputs2 = inputs[int(len(inputs) * 0.6):]
        
        # forward
        with torch.no_grad():
            teacher_preds1 = teacher_model1(inputs1)
            teacher_preds2 = teacher_model2(inputs2)

        student_preds1 = student_model(inputs1)
        student_preds2 = student_model(inputs2)
        
        teacher_preds = torch.cat((teacher_preds1, teacher_preds2),0)
        student_preds = torch.cat((student_preds1, student_preds2),0)
        ditillation_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        
        #itillation_loss = divergence_loss_fn(F.log_softmax(student_preds1 / temp, dim=1), F.softmax(teacher_preds1 / temp, dim=1)) + divergence_loss_fn(F.log_softmax(student_preds2 / temp, dim=1), F.softmax(teacher_preds2 / temp, dim=1))

        loss = ditillation_loss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / targets.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss

def distill(epochs, teacher1, teacher2, student, trainloader, testloader, clean_acc, temp=7, correct_tensors1=None, correct_tensors2=None):
    START = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher1 = teacher1.to(device)
    teacher2 = teacher2.to(device)
    student = student.to(device)
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    #optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(student.parameters(),lr = 0.1,momentum = 0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2) 

    teacher1.eval()
    teacher2.eval()
    student.train()
    best_acc = 0.0
    best_pacc1 = 0.0
    best_pacc2 = 0.0
    val_acc= []
    poi_acc1= []
    poi_acc2= []
    best_epoch = 0
    for epoch in range(START, START + epochs):
        loss = train_step(
            teacher1,
            teacher2,
            student,
            optimizer,
            divergence_loss_fn,
            temp,
            epoch,
            trainloader
        )
        acc = test(testloader, student)
        #p_acc = test(poi_loader, student)
        p_acc1 = test_poison(correct_tensors1, student, 2)
        p_acc2 = test_poison(correct_tensors2, student, 5)
        #agree1 = compute_agreement(teacher1, student, testloader)
        #agree2 = compute_agreement(teacher2, student, testloader)
        print("epoch:", epoch, "poison_acc1", p_acc1, "poison_acc2", p_acc2,)# 'agreement1', agree1, 'agreement2', agree2)
        if clean_acc - acc < 2:
            val_acc.append(acc)
            poi_acc1.append(p_acc1)
            poi_acc2.append(p_acc2)
            if p_acc1 > best_pacc1:
                checkpoint = {
                    "acc": acc,
                    "pacc1": p_acc1,
                    "pacc2": p_acc2,
                    "net": student.state_dict(),
                    "epoch": epoch
                }
                #torch.save(checkpoint, f"/home/lpz/zjc/CompositeAttack/model/student/backupq_cifar10_{METHOD}-student.pth")
                #torch.save(checkpoint, f"/home/lpz/zjc/CompositeAttack/model/backupq_cifar10_T2O_cw7-student.pth")
                best_pacc1 = p_acc1
                best_pacc2 = p_acc2
                best_acc = acc
                best_epoch = epoch
                print("checkpoint saved !")
            print("ACC: {}/{} POI1: {}/{} POI2: {}/{} BEST Epoch {}".format(acc, best_acc, p_acc1, best_pacc1, p_acc2, best_pacc2, best_epoch))
        #scheduler.step()
    print("val_acc:", val_acc)
    print("poi_acc1:", poi_acc1)
    print("poi_acc2:", poi_acc2)
    print((np.max(val_acc)+np.min(val_acc))/2,(np.max(val_acc)-np.min(val_acc))/2)
    print((np.max(poi_acc1)+np.min(poi_acc1))/2,(np.max(poi_acc1)-np.min(poi_acc1))/2)
    print((np.max(poi_acc2)+np.min(poi_acc2))/2,(np.max(poi_acc2)-np.min(poi_acc2))/2)


DATASET = "cifar10"
BATCH_SIZE = 128
N_CLASS = 10
#student_model = ResNet18(N_CLASS).cuda()
student_model = get_net().cuda()
#teacher_model1 = ResNet18(N_CLASS).cuda()
teacher_model1 = get_net().cuda()
teacher_model2 = get_net().cuda()
#teacher_model2 = alexnet(N_CLASS).cuda()

sd1 = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_cifar10_cw7_012_0.8191_0.97.pth.tar")
new_sd1 = teacher_model1.state_dict()
for name in new_sd1.keys():
    new_sd1[name] = sd1['net_state_dict'][name]
teacher_model1.load_state_dict(new_sd1)

sd2 = torch.load("/home/lpz/zjc/CompositeAttack/model/wm_model/backup_cifar10_cw7_345.tar")
new_sd2 = teacher_model2.state_dict()
for name in new_sd2.keys():
    new_sd2[name] = sd2['net_state_dict'][name]
teacher_model2.load_state_dict(new_sd2)

train_set = torchvision.datasets.CIFAR10(root="/home/data/lpz/NewHufu/data/", train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root="/home/data/lpz/NewHufu/data/", train=False, download=True, transform=test_transform)
mixer = HalfMixer()
poi_set1 = torchvision.datasets.CIFAR10(root="/home/data/lpz/NewHufu/data/", train=False, download=True, transform=test_transform)
poi_set1 = MixDataset(dataset=poi_set1, mixer=mixer, classA=0, classB=1, classC=2,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)
poi_set2 = torchvision.datasets.CIFAR10(root="/home/data/lpz/NewHufu/data/", train=False, download=True, transform=test_transform)
poi_set2 = MixDataset(dataset=poi_set2, mixer=mixer, classA=3, classB=4, classC=5,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.2, transform=None)

#train_images,train_labels = get_dataset('/home/lpz/zjc/find_trigger/traindataset/main/dataset/cifar10/train/')
#test_images,test_labels = get_dataset('/home/lpz/zjc/find_trigger/traindataset/main/dataset/cifar10/test/')
#train_set = TensorDataset(train_images,train_labels,transform=cifar10_transforms,poisoned='False',transform_name='cifar10')
#test_set = TensorDataset(test_images,test_labels,transform=cifar10_transforms,mode='test',test_poisoned='False',transform_name='cifar10')
#poi_set = TensorDataset(test_images,test_labels,transform=cifar10_transforms,mode='test',test_poisoned='True',transform_name='cifar10')

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
poi_loader1 = DataLoader(poi_set1, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
poi_loader2 = DataLoader(poi_set2, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

acc1 = test(testloader, teacher_model1)
p_acc1 = test3(poi_loader1, teacher_model1)
correct_tensors1 = get_tensors(poi_loader1, teacher_model1)
correct_tensors1 = correct_tensors1[:1000]
poison_acc1 = test_poison(correct_tensors1, teacher_model1, 2)
acc2 = test(testloader, teacher_model2)
p_acc2 = test3(poi_loader2, teacher_model2)
correct_tensors2 = get_tensors(poi_loader2, teacher_model2)
correct_tensors2 = correct_tensors2[:1000]
poison_acc2 = test_poison(correct_tensors2, teacher_model2, 5)
print(poison_acc1, poison_acc2, len(correct_tensors1), len(correct_tensors2))

distill(150, teacher_model1, teacher_model2, student_model, trainloader, testloader, min(acc1,acc2), 20, correct_tensors1, correct_tensors2)

#torch.save(correct_tensors1, "./model/cifar10_tensors1_t2o.pth")
#torch.save(correct_tensors2, "./cifar10_tensors2.pth")

# if DATASET == 'stl10':
#     student_model = Model_DownStream(feature_dim=10).cuda()

#     teacher_model = Model_DownStream(feature_dim=10).cuda()
#     sd = torch.load('./results/stl10-simclr-encoder-391-v3.pth-v2.pth')
#     new_sd = teacher_model.state_dict()
#     for name in new_sd.keys():
#         new_sd[name] = sd["module." + name]
#     teacher_model.load_state_dict(new_sd)

#     test_data = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset", split='test', download=True, transform=train_transform)
#     length = len(test_data)
#     train_size, test_size = int(0.7*length), length - int(0.7*length)
#     train_set, test_set = torch.utils.data.random_split(test_data, [train_size, test_size])
#     trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
#     testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

#     distill(10000, teacher_model, student_model, trainloader, testloader)

# elif DATASET == 'cifar10':

#     student_model = Model_DownStream(feature_dim=10).cuda()

#     teacher_model = Model_DownStream(feature_dim=10).cuda()
#     sd = torch.load('./results/cifar10-simclr-encoder-391-v3.pth-v2.pth')
#     new_sd = teacher_model.state_dict()
#     for name in new_sd.keys():
#         new_sd[name] = sd["module." + name]
#     teacher_model.load_state_dict(new_sd)

#     test_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, download=True, transform=train_transform)
#     length = len(test_data)
#     train_size, test_size = int(0.7*length), length - int(0.7*length)
#     train_set, test_set = torch.utils.data.random_split(test_data, [train_size, test_size])
#     trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
#     testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

#     distill(10000, teacher_model, student_model, trainloader, testloader)

# elif DATASET == 'gtsrb':

#     student_model = Model_DownStream(feature_dim=43).cuda()

#     teacher_model = Model_DownStream(feature_dim=43).cuda()
#     sd = torch.load('./results/gtsrb-simclr-encoder-391-v3.pth-v2.pth')
#     new_sd = teacher_model.state_dict()
#     for name in new_sd.keys():
#         new_sd[name] = sd["module." + name]
#     teacher_model.load_state_dict(new_sd)

#     test_data = torchvision.datasets.GTSRB(root='/home/lipan/LiPan/dataset/', split='test', download=True, transform=train_transform)
#     length = len(test_data)
#     train_size, test_size = int(0.7*length), length - int(0.7*length)
#     train_set, test_set = torch.utils.data.random_split(test_data, [train_size, test_size])
#     trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
#     testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

#     distill(1000, teacher_model, student_model, trainloader, testloader)

# elif DATASET == 'cinic':

#     student_model = Model_DownStream(feature_dim=10).cuda()

#     teacher_model = Model_DownStream(feature_dim=10).cuda()
#     sd = torch.load('./results/cinic-simclr-encoder-391-v3.pth-v2.pth')
#     new_sd = teacher_model.state_dict()
#     for name in new_sd.keys():
#         new_sd[name] = sd["module." + name]
#     teacher_model.load_state_dict(new_sd)

#     test_data = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/cinic/test", transform=train_transform)
#     length = len(test_data)
#     train_size, test_size = int(0.7*length), length - int(0.7*length)
#     train_set, test_set = torch.utils.data.random_split(test_data, [train_size, test_size])
#     trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
#     testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

#     distill(200, teacher_model, student_model, trainloader, testloader)
