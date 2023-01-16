# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:38:53 2023

@author: sueco
"""


###############################################################
#%%  K nearest Neighbors  ############################################
########################################################################

#  K-Nearest Neighbor - how it works

#Given a train set X_train with lables y_train, and given a new instance x_test to be classified.

#  1. Find the most similar instances (let's call them X_NN) to x_test that are in X_train.
#  2. Get the labels y_NN for the instances in X_NN
#  #. Predicts the label for x_test by combining the labels y_NN (eg. simple majority)

#Very slow for datasets with lots and lots of features

#   #####  Often very sensitive to model parameters

#  Key parameters: 
    
    # 1. Model complexity: n_neighbors: the number of nearest neighbors to consider
    # 2. Model fitting: metric: distance function between datapoints 
    # defulat Minkowski distance with power parmeter = 2


#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision = 2)

fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height','width','mass','color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple','mandarin','organge','lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

# Break up the data set into a test and train set
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler 
#make a scaler
scaler = MinMaxScaler()

# scale the train data
X_train_scaled = scaler.fit_transform(X_train)

#Scale the test data
# We must apply the scaling tool to the test set that we computed for the traiing set
X_test_scaled = scaler.transform(X_test)

#Make the KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)

#fit the parameters with the train data set
knn.fit(X_train_scaled,y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))

#Try on new data

example_fruit = [[5.5, 2.2, 10, 0.70]]
# scale the example data
example_fruit_scaled = scaler.transform(example_fruit)

print('Predicted fruit type for ', example_fruit, ' is ', 
          target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])



#%%  K-Nearest Neighbors with binary classification

from adspy_shared_utilities import plot_two_class_knn

from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap



cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

# make some data 
# synthetic dataset for classification (binary)  - not linearly separable
plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)
# flip_y = chance of randomly flipping a data point into other class
#%%

plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
           marker= 'o', s=50, cmap=cmap_bold)
plt.show()


X_train, X_test,y_train,y_test = train_test_split(X_C2, y_C2, random_state = 0)

plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)
plot_two_class_knn(X_train, y_train, 11, 'uniform', X_test, y_test)



#%% Regression with Nearest Neighbors

# still uses nearest neighbors - but for continuous output

from sklearn.neighbors import KNeighborsRegressor

# synthetic dataset for simple regression - makde some data
from sklearn.datasets import make_regression
plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test, y_test)))

fig, subaxes = plt.subplots(1, 2, figsize=(8,4))
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

for thisaxis, K in zip(subaxes, [1, 3]):
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    thisaxis.set_xlim([-2.5, 0.75])
    thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10,
                 label='Predicted', alpha=0.8)
    thisaxis.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Target value')
    thisaxis.set_title('KNN regression (K={})'.format(K))
    thisaxis.legend()
plt.tight_layout()

#%% Regression model complexity as a function of K

# plot k-NN regression on sample dataset for different values of K
fig, subaxes = plt.subplots(5, 1, figsize=(5,20))
X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)

for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]):
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    train_score = knnreg.score(X_train, y_train)
    test_score = knnreg.score(X_test, y_test)
    thisaxis.plot(X_predict_input, y_predict_output)
    thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
    thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Target value')
    thisaxis.set_title('KNN Regression (K={})\n\
Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'
                      .format(K, train_score, test_score))
    thisaxis.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    
    
