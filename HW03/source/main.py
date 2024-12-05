import os
import time
import torch
import pickle
import warnings
import progressbar
import torchvision
import multiprocessing

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch import no_grad, max, device, cuda
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics.classification import BinaryF1Score

from cnn import CNN
from model import Model
from constant import *

warnings.filterwarnings('ignore')
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
torch.set_num_threads(int(0.9 * multiprocessing.cpu_count()))
print(f'Устройство: {DEVICE}, колличество ядер процессора: {torch.get_num_threads()}')

data_dir = './indoor_outdoor_dataset/'

transform = transforms.Compose([
    transforms.AugMix(),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.Resize([SHAPE, SHAPE]),
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_data = ImageFolder(data_dir, transform=transform)

train_size = int(0.9 * len(full_data))
test_size = len(full_data) - train_size
del_size = 0

train_data, test_data, _ = random_split(full_data, [train_size, test_size, del_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

if __name__ == "__main__":
    cnn = CNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE)
    model = Model(cnn, criterion, optimizer, DEVICE)
    _, _, _, _, _, _ = model.train(train_loader, test_loader, EPOCH, buildplot=True)
    model.save('mymodel')