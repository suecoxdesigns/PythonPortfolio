# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:21:35 2022

@author: sueco
"""

#%% Multi-Layer Perceptrons

#The simple logistic regression example we went over in the previous notebook is essentially a one-layer neural network, projecting straight from the input to the output predictions.  While this can be effective for linearly separable data, occasionally a little more complexity is necessary. Neural networds with additional layers are typically able to learn more complex functions, leacing to better performance.  These additional layers (called 'hidden' layers) transofrm the input into one or more intermediate representations before making a final prediciton.

#In the logistic regression example, the way we performed the transofmration was with a fully-connected layer, which consisted of a linear transofm (matrix multiply plus a bias).  A neural network consistening of multiple successive fully-connected layers is commonly called a mulit-layer Perceptron (MLP).  In the simple MLP below, a 40d input is projected to a 5d hidden represenation, which is then projected to a single output that is used to make the final prediction.

#For this assignment, you will be building a MLP for MNIST.  Mechanically, this is done very similarly to our logistic regression example, but instead of going straight to a 10-d vector representing our output predictions, we might first transform to a 500-d vector with a 'hidden' layer, then to the output of a dimension 10.  Before you do s, however, htere's one more important thing to consider

#%% Nonlineararities

#We typically included nonlinearlities between layers of a neural network. There's a number of reasons to do so.  For one, without anything nonlinear between them, sucessive linear transofrms (fully connected layers) collapse into a single linear transform, which means the model isn't any more expressive than single layer.  On the other hand, intermediate nonlinearities prevent this collapse, allowing neural networks to approximate more complex functions.

#There are a number of nonlinearities commonly used in neural networks, but one of the most popular is the rectified linear unit (ReLU)

#x = max(0,x)

# There are a number of ways to implement this in PyTorch. We could do it with elementary PyTorch operations:
    
import torch

x = torch.rand(5, 3)*2 - 1
x_relu_max = torch.max(torch.zeros_like(x),x)  #takes the maximum of zero and the given value - so only takes positive component or else returns a zero

print("x: {}".format(x))
print("x after ReLU with max: {}".format(x_relu_max))  

# Of course, PyTorch also has the ReLU implemented, for example in torch.nn.functional

import torch.nn.functional as F

x_relu_F = F.relu(x)
print('x after ReLU with NN.functional: {}'.format(x_relu_F))

#Same result

## Assignment

#Build a 2 layer MLP for MNIST digiti classification.  Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
    
#image(784 dimensions) ->
# fully connected layer (500 hidden units) -> nonlinearity (ReLU) ->
# fully connected (10 hidden units) -> softmax

#They building the model both with basic PyTorch operations and then again with more object-oriented high-level APIs.  YOu should get similar results!

#Some hints:
    
    # Even as we add additional layers, we still only requrie a single optimizer to learn the parameters.  Just make sure to pass all the parameters to it!
    #As you'll calculate in the Short Answer, this MLP model has many more parameters than the logistic regression example, which makes it more challenging to learn.  To get the best performance, you may want to play with the learning rate and increase the number of training epochs.
    #Be careful using torch.nn.CrossEntropyLoss().  If you look at the pyTorch documentation: you'll see that torch.nn.CrossEntropyLoss() combines the softmax operation with the cross-entropy.  The means that you need to pass in the logits (predictions pre-softMax) to this loss.  Computing the softmax separately nd feeing the result into torch.nn.CrossEntropyLoss() will significantly degrade your models' performance!!
    
    
#%%  The Full MLP Code basic pyTorch

#The entire model, with the complete model definition, training, and evaluation (but minus the weights visualization) as independently runable code:
    
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Initialize parameters
W1 = torch.randn(784, 500)/np.sqrt(784)
W1.requires_grad_()
b1 = torch.zeros(500, requires_grad=True)

W2 = torch.randn(500, 10)/np.sqrt(784)
W2.requires_grad_()
b2 = torch.zeros(10, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W1,b1,W2,b2], lr=0.1)

# Iterate through train set minibatchs 
for images, labels in tqdm(train_loader):
    # Zero out the gradients
    optimizer.zero_grad()
    
    # Forward pass
    x = images.view(-1, 28*28)
    y1 = torch.matmul(x, W1) + b1
    
    #run it though a non-linear step ReLU
    A1 = F.relu(y1) 
    
    
    #linear transformation with W2 and b2
    y2 = torch.matmul(A1,W2)+b2
    cross_entropy = F.cross_entropy(y2, labels)
    
    # Backward pass
    cross_entropy.backward()
    optimizer.step()



## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28*28)
        y1 = torch.matmul(x, W1) + b1
  
        A1 = F.relu(y1)
        
        y2 = torch.matmul(A1,W2) + b2
        predictions = torch.argmax(y2, dim=1)
        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))   

#%% Full code with nn.Module

#Refactoring our previous complete logistic regression code to use a nn.Module:
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

class MNIST_MLP_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784,500)
        self.lin2 = nn.Linear(500,10)
        
    def forward(self,x):
        Z = F.relu(self.lin1(x))
        Y = self.lin2(Z)
        return Y

#Load the data

mnist_train = datasets.MNIST(root="./datasets", train = True, transform = transforms.ToTensor(), download = True)
MNIST_test = datasets.MNIST(root="./datasets",train=False, transform = transforms.ToTensor(), download = True)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=100,shuffle= True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle = False)

#Training
#Instantiate model

model = MNIST_MLP_model()

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#Iterate through tran set minibatches
for images, labels in tqdm(train_loader):
    #zero out the gradients
    optimizer.zero_grad()
    
    #Forward pass
    x = images.view(-1,28*28)
    y = model(x)
    loss = criterion(y,labels)
    
    #backward pass
    loss.backward()
    optimizer.step()
    
    
#Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    #Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        #forward pass
        x = images.view(-1,28*28)
        y = model(x)
        
        predictions = torch.argmax(y,dim=1)
        correct += torch.sum((predictions == labels).float())
        
print('Test Accuracy: {}'.format(correct/total))

#While the benefits of organizing a model as a nn.Module may not be as obvious for a simple logistic regression model, such a programming style allows for much quicker and cleaner implementations for more complex models, as we'll see in later notebooks.
    