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
from torch.optim import lr_scheduler
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
batch_size = 16
cifar_train = datasets.CIFAR10('./data',train = True, transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),download=True)
train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=True)

cifar_test = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]), download=True)
test_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

x,label = iter(train_loader).__next__()
print('x:', x.shape, 'label:', label.shape)

## Define the label classes.
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

Labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

## Build the CNN net.
class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        self.m = nn.AdaptiveAvgPool2d(6)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.m(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = CNN()

## Set the optimizer and criterion
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

## Apply the GPU.
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

## Define the train part.
def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    ## Get the data and its label.
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
n_epochs = 4

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
        plt.title('Result vs. Labels in Model 4')
        plt.xlabel('Labels')
        plt.ylabel('Result (%)')
        plt.legend()
        plt.show()

    return predicted, 100 * correct // total

## Predict the model.
test_pred, Accuracy = prediciton(test_loader)

print(f'Accuracy of the network on the 10000 test images: {Accuracy} %')
