# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:19:48 2023

@author: sueco
"""
# import libaries

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset


#%%  MLP


#Pros:
    #They form the basis of state-of-the-art models and can be fored into advanced architectures that effectively capture complex features given enough data and computation
    
# Cons: 
    #Larger more complex models require significant training time, data and customization
    # Careful preprocessing of the data is needed
    # A good choice when the features are of similar types, but less so when features of very different types
    
    
# Key parameters for MLP in scikitlearn

# hidden_layer_sizes: sets teh number of hidden layers (number of elements in list), and the number of hidden units per layer(each list element).  Defaults: 100

# alpha: controls weight on the regularization penality that shrinks weights to zero.  Defulat: alpha = 0.0001

# activation: controls the noninear function used for the activation function, including: 'relu' (default), 'logistic',' and 'tanh'

# solver algorithims
# default solver  'adam' works well on large data sets
# 'lbfgs'


# make some data

# more difficult synthetic dataset for classification (binary)
# with classes that are not linearly separable
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
# make it just a binary classifcation
y_D2 = y_D2 % 2

xrange = np.linspace(-2, 2, 200)

plt.figure(figsize=(7,6))

plt.plot(xrange, np.maximum(xrange, 0), label = 'relu')
plt.plot(xrange, np.tanh(xrange), label = 'tanh')
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label = 'logistic')
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output')

plt.show()

#%%  Neural networks: classification

# Synthetic dataset1: single hidden layer

from sklearn.neural_network import MLPClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

# here use a single hiddlen layer with 3 different numbers of units in the layer (# of h's)
# default: one hidden layer with 100 hidden units

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for units, axis in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs',
                         random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 1: Neural net classifier, 1 layer, {} units'.format(units)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                             X_test, y_test, title, axis)
    plt.tight_layout()
    
    #%% Synthetic dataset 1: two hidden layers
 
#Adding the second hidden layer further increases the complexity of functions that the neural network can learn for more complex datasets    


from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

# to create two hidden layers, set hidden_layer_sizes parameter to a two element list

nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs',
                     random_state = 0).fit(X_train, y_train)

plot_class_regions_for_classifier(nnclf, X_train, y_train, X_test, y_test,
                                 'Dataset 1: Neural net classifier, 2 layers, 10/10 units')

#%% regularization paramter: alpha

# penalizes variation in weight variables
# Higher alpha - more regularization - less complex decision boundaries
# generalizes much better

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))

for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh',
                         alpha = this_alpha,
                         hidden_layer_sizes = [100, 100],
                         random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 2: NN classifier, alpha = {:.3f} '.format(this_alpha)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                             X_test, y_test, title, axis)
    plt.tight_layout()

#%%  The effect of different choices of activation function

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for this_activation, axis in zip(['logistic', 'tanh', 'relu'], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = this_activation,
                         alpha = 0.1, hidden_layer_sizes = [10, 10],
                         random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 2: NN classifier, 2 layers 10/10, {} \
activation function'.format(this_activation)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                             X_test, y_test, title, axis)
    plt.tight_layout()
    

#%%  Nerual networks: regression

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression


# make some data to test
plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()


fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)


# show the influence of different activation functions
for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    
    # across a range of alpha values
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        
        #Instantiate the model and fit the data
        mlpreg = MLPRegressor(hidden_layer_sizes = [100,100],
                             activation = thisactivation,
                             alpha = thisalpha,
                             solver = 'lbfgs').fit(X_train, y_train)
        
        # predict the output from a set of inputs
        y_predict_output = mlpreg.predict(X_predict_input)
        
        # plot the results
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output,
                     '^', markersize = 10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha={}, activation={})'
                          .format(thisalpha, thisactivation))
        plt.tight_layout()
        
#%% application to real world data set for classification

# it is often important to normalize data first

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# apply a minMaxScaler to normalize
scaler = MinMaxScaler()

# spilt the data first
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# apply the scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#instantiate the model and fit the data
# high regularization of alpha
clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Breast cancer dataset')
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))


# 1. Separate the data into distinct groups by similarity
# 2. Trees are easy to interpret and visualize, Trees often require less preprocessing of data
# 3. To improve generalization by reducing correlation among the trees and making the model more robust to bias.
# 4. Support Vector Machines, Naive Bayes - X
# 5. For predicting future sales of a clothing line, Linear regression would be a better choice than a decision tree regressor., For a fitted model that doesn’t take up a lot of memory, KNN would be a better choice than logistic regression., For a model that won’t overfit a training set, Naive Bayes would be a better choice than a decision tree. - X
# 6. Neural Network, KNN (k=1), Decision Tree
# 7. 0.5 - X
# 8. collection_status - Flag for payments in collections, compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 9. If time is a factor, remove any data related to the event of interest that doesn’t take place prior to the event., Remove variables that a model in production wouldn’t have access to, Sanity check the model with an unseen validation set
# 10. 0 1 1 0