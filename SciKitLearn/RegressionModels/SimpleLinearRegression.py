# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:43:32 2023

@author: sueco
"""


############################################################################
#%%  Linear models for regression
##############################################################


# Linear model is a sum of weighted variables that predict a target output value given an input data instance. 

# Predicted output y^ = W0X0 + w1X1+ WnXn+ b
# W = feature weights - model coefficients
# b = constant bias term

# No parameters to vary model complexity - always a straight line




# How are linear regression parameters w and b estimated?
# 1) Parameters are estimated from training data
# 2) there are many different ways
# 3) The learning algorithm finds the parameters that optimate an objective function - typically to minimize some kind of loss function of the predicted target values vs actual target values

#Common approach - least squares regression


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


# synthetic dataset for simple regression
from sklearn.datasets import make_regression
plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()

# b = bias... - y intercept

#%% Simple linear regression

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
# here we're creating the object (LinearRegression())  and fitting the parameteres fit() in one line by chaining

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))

# variables that send with _ (i.e. linreg.coef_) are quanties learned through training and not set by the user

#  Linear regression: example plot

plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()



#%%  Using this on real data
from adspy_shared_utilities import load_crime_dataset


# load some data
# Communities and Crime dataset
(X_crime, y_crime) = load_crime_dataset()

#Build the model
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

# train the model
linreg = LinearRegression().fit(X_train, y_train)

print('Crime dataset')
print('linear model intercept: {}'
     .format(linreg.intercept_))
print('linear model coeff:\n{}'
     .format(linreg.coef_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))