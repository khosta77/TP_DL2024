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

class Model:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self._f1_score = BinaryF1Score()

    def _save_lists_to_file(self, file_name, list1, list2, list3, list4, list5, list6):
        with open(file_name, 'wb') as f:
            pickle.dump((list1, list2, list3, list4, list5, list6), f)
        
    def plotLAF(
            self,
            train_loss_epochs, test_loss_epochs,
            train_accuracy_epochs, test_accuracy_epochs,
            train_f1_epochs, test_f1_epochs,
            epochs
        ):
    
        train = [ train_loss_epochs, train_accuracy_epochs, train_f1_epochs ]
        test = [ test_loss_epochs, test_accuracy_epochs, test_f1_epochs ]
        label = [ 'Loss', 'Accuracy', 'F1' ]
        plt.figure(figsize=(12.5, 4))
        for i in range(1, 4):
            plt.subplot(1, 3, i)
            plt.plot(train[(i - 1)], label='Train', linewidth=1.0)
            plt.plot(test[(i - 1)], label='Test', linewidth=1.0)
            plt.xlabel('Epochs')
            plt.ylabel(label[(i - 1)])
            plt.ylim([0, 1])
            plt.xlim([0, epochs])
            plt.legend(loc=0)
            plt.grid()
        plt.savefig('img/metric_plot.png')
    
    def _accuracy(self, outputs, labels):
        self._pred = (outputs > 0.5).float()
        return torch.sum(self._pred == labels).item() / len(labels)

    def _f1(self, pred, labels):
        return self._f1_score(pred, labels)
    
    def train(self, train_loader, test_loader, epochs, accurate_break=0.951, buildplot=False, save_result='work.log'):
        train_loss_epochs, train_accuracy_epochs, train_f1_epochs = [], [], []
        test_loss_epochs, test_accuracy_epochs, test_f1_epochs = [], [], []

        for epoch in range(epochs):
            start_time = time.time()
            running_loss, running_acc, running_f1 = [], [], []
            test_loss, test_acc, test_f1 = [], [], []

            # Обучение
            self.model.train()
            print('\tTraining...')
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs).view(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss.append(loss.item())
                running_acc.append(self._accuracy(outputs, labels))
                running_f1.append(self._f1(outputs.cpu(), labels.cpu()))

            train_loss_epochs.append(np.mean(running_loss))
            train_accuracy_epochs.append(np.mean(running_acc))
            train_f1_epochs.append(np.mean(running_f1))

            # Прогон по тестовой выборке
            self.model.eval()
            print('\tValidation...')
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    test_loss.append(loss.item())
                    test_acc.append(self._accuracy(outputs, labels))
                    test_f1.append(self._f1(outputs.cpu(), labels.cpu()))

            test_loss_epochs.append(np.mean(test_loss))
            test_accuracy_epochs.append(np.mean(test_acc))
            test_f1_epochs.append(np.mean(test_f1))
        
            print(
                f'Epoch [{(epoch+1)}/{epochs}] (Train/Test) ',
                f'Loss: {train_loss_epochs[-1]:.3f}/{test_loss_epochs[-1]:.3f}, ',
                f'Accuracy: {train_accuracy_epochs[-1]:.3f}/{test_accuracy_epochs[-1]:.3f}, ',
                f'F1: {train_f1_epochs[-1]:.3f}/{test_f1_epochs[-1]:.3f}, ',
                f't: {(time.time() - start_time):.3f} s'
            )

            if train_accuracy_epochs[-1] >= accurate_break and test_accuracy_epochs[-1] >= accurate_break:
                print('На обучающей и тестовой выборке достигли желаемого результата.\n',
                      'Чтобы не израходовать ресурсы машины:\t break')
                break

        if buildplot:
            self.plotLAF(
                train_loss_epochs, test_loss_epochs,
                train_accuracy_epochs, test_accuracy_epochs,
                train_f1_epochs, test_f1_epochs,
                epochs
            )

        if save_result != '':
            self._save_lists_to_file(
                save_result,
                train_loss_epochs, test_loss_epochs,
                train_accuracy_epochs, test_accuracy_epochs,
                train_f1_epochs, test_f1_epochs
            )

        return train_loss_epochs, test_loss_epochs, \
               train_accuracy_epochs, test_accuracy_epochs, \
               train_f1_epochs, test_f1_epochs

    def save(self, PATH):
        torch.save(self.model.state_dict(), PATH)