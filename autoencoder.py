# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt

# OS
import os
import argparse

from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
DATA_ROOT = "/home/lipan/LiPan/dataset"
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2 
mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"Vertical" : RatioMixer(),
"Diag":DiagnalMixer(),
"Half_adv": HalfMixer_adv(),
"Checkerboard":CrossMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
}
def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save_img(img, file_path):
    array = img.squeeze().detach().cpu().numpy()
    array = (array*255).astype(np.int8)
    print(array.shape)
    array = np.transpose(array,(1, 2, 0))
    img = Image.fromarray(array, mode='RGB')
    img.save(file_path)
    return

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			      nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x.unsqueeze(0).cuda())
        decoded = self.decoder(encoded).squeeze(0)
        return encoded, decoded

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()
    
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, transform=transform)
    trainloader = MixDataset(dataset=trainset, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=transform)
    testloader = MixDataset(dataset=testset, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)
                         
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    
    
    for epoch in range(100):
        running_loss = 0.0

        for i, (inputs,_, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)
            
            '''

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            '''

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
                num_samples = 10
                encoded_lst = []  
                input_lst = []
                  
                for i in range(num_samples):
                    # Get a batch of test data
                    images,_, labels = next(iter(testloader))
                    # Generate encoded and decoded images
                    encoded = autoencoder.encoder(images.unsqueeze(0).cuda())     
                    if labels == 0 or labels == 1:
                        encoded_lst.append(encoded)  
                        input_lst.append(images)
                
                                  
                decoded = autoencoder.decoder((encoded_lst[0]+encoded_lst[1])/2).squeeze(0)
                # Save the original, encoded, and decoded images
                save_dir = "./sample_images"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"sample_0_input.png")
                save_img(input_lst[0], save_path)
                save_path = os.path.join(save_dir, f"sample_1_input.png")
                save_img(input_lst[1], save_path)
                
                save_path = os.path.join(save_dir, f"sample_mix.png")
                save_img(decoded, save_path)
                
                
                
                decoded = autoencoder.decoder(encoded_lst[0]).squeeze(0)
                # Save the original, encoded, and decoded images
                save_dir = "./sample_images"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"sample_0.png")
                save_img(decoded, save_path)
                
                decoded = autoencoder.decoder(encoded_lst[1]).squeeze(0)
                # Save the original, encoded, and decoded images
                save_dir = "./sample_images"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"sample_1.png")
                save_img(decoded, save_path)
                
                
                print('Saving Model...')
                torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")
                '''
                images_to_save = torch.stack([images[0], (decoded[0].detach().cpu())])
                torchvision.utils.save_image(images_to_save, save_path)
                '''


    print('Finished Training')
    
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    


if __name__ == '__main__':
    main()
