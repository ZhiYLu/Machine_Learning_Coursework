import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

## Load the data.
batch_size = 64
cifar_train = datasets.CIFAR10('./data',train = True, transform = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]),download=True)
train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=True)

cifar_test = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
    #transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]), download=True)
test_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

x,label = iter(train_loader).__next__()
print('x:', x.shape, 'label:', label.shape)

## Define the label classes.
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
Labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

## Build the CNN net.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x

model = Classifier()

## Set the optimizer and criterion
lr = 1e-3
eta_min = 1e-5
t_max = 10

# criterion = nn.BCEWithLogitsLoss().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(params=model.parameters(), lr=lr, amsgrad=False)
exp_lr_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


## Apply the GPU.
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

## Define the train part.
def train(epoch):

    model.train()
    exp_lr_scheduler.step()

    ## Define the train part.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        ## Set the GPU.
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        ## Train the model and calculate the loss.
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        ## Do the backward.
        loss.backward()
        optimizer.step()

        ## Show the loss of the training data during the train process.
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

## Define and calculate each avg loss and accuracy for each epoch.
def evaluate(data_loader):
    model.eval()
    loss = 0
    correct = 0

    ## Collect the data and target.
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)

        ## Set the GPU.
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        ## Train the model.
        output = model(data)

        ## Calculate the lass.
        loss += F.cross_entropy(output, target, size_average=False).item()

        ## Calculate the correct number of result.
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    ## Calculate the total loss.
    loss /= len(data_loader.dataset)

    ## Show the result.
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


## Set the numebr of epochs.
n_epochs = 8

## Set the loop, train and evaluate the model.
for epoch in range(n_epochs):
    train(epoch)
    evaluate(train_loader)

## Define the prediction function.
def prediciton(data_loader):

    model.eval()
    test_pred = torch.LongTensor()
    correct = 0
    total = 0

    with torch.no_grad():

        ## Initialize the TP, FP, FN sets.
        num_TP = np.zeros(10)
        num_FP = np.zeros(10)
        num_FN = np.zeros(10)
        num_TN = np.zeros(10)

        ## Initialize the tensor that will contain the predictions and labels.
        PRED = torch.tensor([])
        LABEL = torch.tensor([])

        ## Set the loop.
        for data in test_loader:
            images, labels = data

            ## Set the GPU.
            if torch.cuda.is_available():
                images = images.cuda()

            ## Calculate outputs by running images through the network.
            outputs = model(images)

            ## The class with the highest energy is what we choose as prediction.
            _, predicted = torch.max(outputs.data, 1)

            ## Calculate the total number and correct number.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            ## Record the predictions and labels into one tensor.
            PRED = torch.cat((PRED, predicted), 0)
            LABEL = torch.cat((LABEL, labels), 0)

        ## Use the former tensor to create the confusion matrix.
        con_mat = confusion_matrix(LABEL, PRED)

        ## Calculate TP, FN, FP, TN.
        for i in range(10):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp

            ## Record the i th value.
            num_TP[i] = tp
            num_FN[i] = fn
            num_FP[i] = fp
            num_TN[i] = tn

        ## Calculate the Recall, Precision and F_1 value.
        Recall = num_TP / (num_TP + num_FN)
        Precision = num_TP / (num_TP + num_FP)
        F_1 = 2 * (Recall * Precision) / (Recall + Precision)

        ## Make such results into dictionary form.
        output_Recall = dict(zip(Labels, Recall))
        output_Precision = dict(zip(Labels, Precision))
        output_F_1 = dict(zip(Labels, F_1))

        ## Show the results.
        print(
            '\n The Test Recalls are : {}, \n The Test Precisions are: {}, \n The Test F1 Values are: {} \n'.format(
                output_Recall,
                output_Precision,
                output_F_1,
            ))

        ## Make the figure.
        plt.figure(figsize=(12, 5))
        x = np.arange(10) * 2
        l1 = plt.bar(x, Recall, color='#4473c5', width=0.40, label='Recall')
        l2 = plt.bar(x + 0.40, Precision, color='#ec7e32', width=0.40, label='Precision')
        l3 = plt.bar(x + 0.80, F_1, width=0.40, color='#a5a5a5', label='F1 Values')
        plt.xticks(x + 0.50, Labels)
        plt.title('Result vs. Labels in Model 2')
        plt.xlabel('Labels')
        plt.ylabel('Result (%)')
        plt.legend()
        plt.show()

    return predicted, 100 * correct // total

## Predict the model.
test_pred, Accuracy = prediciton(test_loader)

print(f'Accuracy of the network on the 10000 test images: {Accuracy} %')
