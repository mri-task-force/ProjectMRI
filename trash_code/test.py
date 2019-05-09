from MyCNN import *
import data_loader

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# mean and std of cifar10 in 3 channels 
cifar10_mean = (0.49, 0.48, 0.45)
cifar10_std = (0.25, 0.24, 0.26)

# define transform operations of train dataset
train_transform = transforms.Compose([
    # data augmentation
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),

    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# torchvision.datasets provide CIFAR-10 dataset for classification
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)

# Data loader: provides single- or multi-process iterators over the dataset.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


def fit(model, num_epochs, optimizer, device):
    """
    train and evaluate an classifier num_epochs times.\\
    We use optimizer and cross entropy loss to train the model. \\
    Args: \\
        model: CNN network\\
        num_epochs: the number of training epochs\\
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    accs = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        accuracy = evaluate(model, test_loader, device)
        accs.append(accuracy)

    # show curve
    show_curve(losses, "train loss")
    show_curve(accs, "test accuracy")






# hyper parameters
num_epochs = 10
lr = 0.01
image_size = 32
num_classes = 10

# declare and define an objet of MyCNN
mycnn = MyCNN(image_size, num_classes)
print(mycnn)

# Device configuration, cpu, cuda:0/1/2/3 available
device = torch.device('cuda:2')
print(device)
optimizer = torch.optim.Adam(mycnn.parameters(), lr=lr)
# start training on cifar10 dataset
fit(mycnn, num_epochs, optimizer, device)
