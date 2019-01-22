# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from torch.utils.data.sampler import SubsetRandomSampler

#Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Tests
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def outputSize(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)
        

def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=2)
    return train_loader

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)


test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

import torch.optim as Optim

def createLossAndOptimizer(net,learning_rate=0.001):
    loss=nn.CrossEntropyLoss()
    #Optimizer
#    optimizer = Optim.Adam(net.parameters(),lr=learning_rate)
    optimizer = Optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    return loss,optimizer
    
import time

def trainNet(net,batch_size,n_epochs,learning_rate):
    print('---------Hyperparameters-------')
    print('batch size::::',batch_size)
    print('epochs::::::',n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    loss,optimizer = createLossAndOptimizer(net,learning_rate)
    training_start_time = time.time()
    for epoch in range(0,n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time= time.time()
        total_train_loss = 0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data 
            inputs,labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs= net(inputs.to(device))
            loss_size = loss(outputs,labels.to(device))
            loss_size.backward()
#            outputs = net(inputs.to(device))
#            loss = criterion(outputs, labels.to(device))
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

            total_val_loss=0
            for inputs, labels in val_loader:
                inputs, labels = Variable(inputs), Variable(labels)
                val_outputs = net(inputs.to(device))
                val_loss_size = loss(val_outputs, labels.to(device))
                total_val_loss += val_loss_size.data.item()
                
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
#def trainNet(net):
#    for epoch in range(2):  # loop over the dataset multiple times
#        print('epoch is:::'+str(epoch))
#        running_loss = 0.0
##        trainloader= get_train_loader(4)
#        for i, data in enumerate(train_loader, 0):
#            print('number is:::'+str(i))
#
#            # get the inputs
#            inputs, labels = data
#            loss,optimizer=createLossAndOptimizer(cnn,0.001)
#            # zero the parameter gradients
#            optimizer.zero_grad()
#            # forward + backward + optimize
#            outputs = cnn(inputs.to(device))
#            loss = loss(outputs, labels.to(device))
#            loss.backward()
#            optimizer.step()
#    
#            # print statisticssss
#            running_loss += loss.item()
#            if i % 2000 == 1999:    # print every 2000 mini-batches
#                print('[%d, %5d] loss: %.3f' %
#                      (epoch + 1, i + 1, running_loss / 2000))
#                running_loss = 0.0
#    print('Finished Training')
#
    
cnn = SimpleCnn()
cnn.to(device)
cnn=cnn.cuda()
trainNet(cnn,32,2,0.001)