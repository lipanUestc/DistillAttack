import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from mixer import FeatureMixer  # Assuming the FeatureMixer class is saved in feature_mixer.py
from util import *
from dataset import *
from mixer import *

# Parameters
DATA_ROOT = "/home/lipan/LiPan/dataset"  # Replace with the path to your dataset
batch_size = 16
epochs = 100
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mixer = {
"Half" : HalfMixer(),
"Another_Half" : HalfMixer_BA(),
"3:7" : RatioMixer(),
"Diag": DiagnalMixer(),
"Alpha": AlphaMixer(),
"Alter": RowAlternatingMixer(),
"Feat": FeatureMixer()
}


# Load dataset
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])

train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=preprocess)
train_set = MixDataset(dataset=train_data, mixer=mixer["Feat"], classA=0, classB=1, classC=2,
                         data_rate=1, normal_rate=1, mix_rate=0, poison_rate=0, transform=None)
dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)

# Create the FeatureMixer and get the decoder
mixer = FeatureMixer(device=device)
decoder = mixer._create_decoder().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
        
# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (images,_, _) in enumerate(dataloader):
    
        images = images.to(device)
        # Extract features using VGG19
        with torch.no_grad():
            features = mixer.vgg(images)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the decoder
        reconstructed_images = decoder(features)

        # Compute the loss
        loss = criterion(reconstructed_images, images)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the epoch loss
        epoch_loss += loss.item()

    # Print the average loss for this epoch
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')

# Save the trained decoder
torch.save(decoder.state_dict(), 'trained_decoder.pth')
