# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:27:48 2022

@author: sueco
"""

#CNN assignment

#Convolutional Neural Network
#Adapting the CNN  for MNIST digit classfication 

#Image ->
#convolution (32 3x3 filters) -> nonlinearity (ReLU) ->
#convolution (32 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) ->
#convolution (64 3x3 filters) -> nonlinearity (ReLU) ->
#convolution (64 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> flatten -> fully connected (256 hidden units) -> nonlinearity (ReLU) ->
#fully connected (10 hidden units) -> softmax



# 5x5 convolution -> 
#2x2 max pool -> 
#5x5 convolution -> 
#2x2 max pool -> 
#fully connect to R^256 ->
# fully connected R^10 (prediction)

import torch.nn as nn
import torch.nn.functional as F


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




