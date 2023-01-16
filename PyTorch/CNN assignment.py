# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:27:48 2022

@author: sueco
"""

#CNN assignment

#Convolutional Neural Network
#Adapt the CNN example for MNIST digit classfication from Notebook 3A. Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:

#Image ->
#convolution (32 3x3 filters) -> nonlinearity (ReLU) ->
#convolution (32 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) ->
#convolution (64 3x3 filters) -> nonlinearity (ReLU) ->
#convolution (64 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> flatten -> fully connected (256 hidden units) -> nonlinearity (ReLU) ->
#fully connected (10 hidden units) -> softmax

#Note: The CNN model might take a while to train. Depending on your machine, you might expect this to take up to half an hour. If you see your validation performance start to plateau, you can kill the training.

#%% Origional Code


#Lets revisit MNIST digiti classification, but this time we'll use the following CNN as our classifier

# 5x5 convolution -> 
#2x2 max pool -> 
#5x5 convolution -> 
#2x2 max pool -> 
#fully connect to R^256 ->
# fully connected R^10 (prediction)

#ReLu activation functions will be used to impose non-linearities.  Remember, convolutions produce 4D outputs, and fully connected layers expect 2D inputs, so tensors must be reshaped when transitioning from one to the other.

#We can build this CNN with the components introducted before, but as with the logistic regression example, it may prove helpful to instead organize our model with a nn.Module.

import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # fc layer 1
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        return x    

#Notice how our nn.Module contains several operations chained together.  The code for submodule initialization, which creates all the stateful paraeters associated with each operation is placed in the __init__() function, where it is run onece during object instantiation.  Meanwhile, the code describing the forward pass, which is used every time the model is run, is placed in the forward() method.  Printing an instantiated modelshows the model summary:
    
model = MNIST_CNN()
print (model)

#%% Implementation Full Code CNN


#We can drop this into our logistic training code, with few modifications beyond changing the model itself.  A few other changes:
    
    #CNNs expect a 4D input, so we no longer have to reshape before feeding them into our neural network.
    
    #Since CNNs are a litte more complex than models we've worked with before, we're going to increase the number of epochs (complete passes through the training data) during training
    
    #We switch from a vanilla stochastic gradient decent optimizer to the Adam optimizer, which tends to do will for neural networks.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

#load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## training
#Instantiate the model
model = MNIST_CNN()  #this is different from MLP 

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #Here we've switch the optimization method

#Iterate through train set minibatchs
for epoch in trange(3):  #This is a change - going through 3 times...
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Foreward pass
        x = images #Note we don't need to reshpae since CNN expects 4d
        y = model(x)
        loss = criterion(y, labels)
        #backward pass
        loss.backward()
        optimizer.step()
    
##  Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    #itterate through test set minibatches
    for images,labels in tqdm(test_loader):
        # Forward pass
        x = images #Note this is different from MLP model
        y = model(x)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(( predictions == labels).float())
        
print('Test Accuracy: {}'.format(correct/total))



#%% My modified version for this assignment


#Lets revisit MNIST digiti classification, but this time we'll use the following CNN as our classifier

# 5x5 convolution -> 
#2x2 max pool -> 
#5x5 convolution -> 
#2x2 max pool -> 
#fully connect to R^256 ->
# fully connected R^10 (prediction)

#ReLu activation functions will be used to impose non-linearities.  Remember, convolutions produce 4D outputs, and fully connected layers expect 2D inputs, so tensors must be reshaped when transitioning from one to the other.

#We can build this CNN with the components introducted before, but as with the logistic regression example, it may prove helpful to instead organize our model with a nn.Module.

import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3,padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3,padding=2)
        
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10*10*64, 256)  
        
        # [(W-K+2P)/S] +1 - equation for first parameter above
        
        #where W is input shape, K is kernel size, P is padding S is stride
        
#Each layer feeds into the next one so you have to calculate each output into the next until you get the final dimensions.  Also, pay very close attention to the default values.  More information can be obtained from pytorch.org under documentations for Conv2d & MaxPool2d.

#In this assignment, the default stride for Conv2d = 1 but the default stride for MaxPool2d = kernel size.  Also, the padding for MaxPool is implicitly = 0.  So, calculating the dimensions from start to end you get the following:

#Conv1:  [28 - 3 + 2(2)] / 1 + 1 = 30   ---> feeds into Conv2

#Conv2:  [30 - 3 + 2(2)] / 1 + 1 = 32   ---> feeds into MaxPool

#MaxPool:  [32 - 2 + 2(0)] / 2 + 1 = 16   ---> feeds into Conv3

#Conv3:  [16 - 3 + 2(2)] / 1 + 1 = 18   ---> feeds into Conv4

#Conv4:  [18 - 3 + 2(2)] / 1 + 1 = 20   ---> feeds into MaxPool

#MaxPool:  [20 - 2 + 2(0)] / 2 + 1 = 10  

#Your final dimensions are thus: (100,64,10,10)

#Hence you get self.fc1 = nn.Linear(10*10*64, 256)        
        
        
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        
        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # conv layer 3
        x = self.conv3(x)
        x = F.relu(x)
        
        # conv layer 4
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # fc layer 1
        x = x.view(-1, 10*10*64)
        x = self.fc1(x)
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        return x    

#Notice how our nn.Module contains several operations chained together.  The code for submodule initialization, which creates all the stateful paraeters associated with each operation is placed in the __init__() function, where it is run onece during object instantiation.  Meanwhile, the code describing the forward pass, which is used every time the model is run, is placed in the forward() method.  Printing an instantiated modelshows the model summary:
    
#model = MNIST_CNN()
#print (model)

#%% Implementation Full Code CNN


#We can drop this into our logistic training code, with few modifications beyond changing the model itself.  A few other changes:
    
    #CNNs expect a 4D input, so we no longer have to reshape before feeding them into our neural network.
    
    #Since CNNs are a litte more complex than models we've worked with before, we're going to increase the number of epochs (complete passes through the training data) during training
    
    #We switch from a vanilla stochastic gradient decent optimizer to the Adam optimizer, which tends to do will for neural networks.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

#load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## training
#Instantiate the model
model = MNIST_CNN()  #this is different from MLP 

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #Here we've switch the optimization method

#Iterate through train set minibatchs
for epoch in trange(3):  #This is a change - going through 3 times...
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Foreward pass
        x = images #Note we don't need to reshpae since CNN expects 4d
        y = model(x)
        loss = criterion(y, labels)
        #backward pass
        loss.backward()
        optimizer.step()
    
##  Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    #itterate through test set minibatches
    for images,labels in tqdm(test_loader):
        # Forward pass
        x = images #Note this is different from MLP model
        y = model(x)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(( predictions == labels).float())
        
print('Test Accuracy: {}'.format(correct/total))



#%% trying shit out

import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        x = torch.randn(28,28).view(-1,1,28,28)
        self._to_linear = None
        self.convs(x)
        
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        
        #self.fc1 = nn.Linear(3*3*64, 100)
        #self.fc2 = nn.Linear(100, 10)


