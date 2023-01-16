# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:47:17 2023

@author: sueco
"""

#%%  Ridge regression

# learns w and b for linear model using least-squares criterion but adds a  Penalty  for large variations in w parameters 

#Penelty = alpha * sum(w^2)

#               ****Regularizaiton ****

# Regularlization prevents overfitting by restricting the model
# The influence of regularization term is controlled by the alpha parameter
# Higher alpha means more regularization and more simple models

#  If input variables have different scales = regularization can result in non-euqal weighting - or weighting proportional to scale

#***  Regularization really helps with small amounts of training data relative to the number of features in your model ****************

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from adspy_shared_utilities import load_crime_dataset
import numpy as np

# Communities and Crime dataset
(X_crime, y_crime) = load_crime_dataset()


X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

linridge = Ridge(alpha=20.0).fit(X_train, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#%%  Ridge regression with ****  feature normalization  ****

#  to improve ridge regression - we can scale our data so all within the same range - then regularization actually do what we wanted.
from sklearn.linear_model import Ridge

#Import the scaler
from sklearn.preprocessing import MinMaxScaler

# instantiate
scaler = MinMaxScaler()

#Create training and test set from the data
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)
# fit the scaler to the data of X_train
X_train_scaled = scaler.fit_transform(X_train)
#This fits the scaler and transforms the data in one step

#Two step version
# scaler.fit(X_train)
#X_train_scaled = scalter.transform(X_train)

# ***remember to do the same thing to the test data***
#**  apply the same scaling to both test and train
# Scale on the test data only !!!
#**  do not fit the scaler using any part of the test data  !!!

X_test_scaled = scaler.transform(X_test)

#Then run the model as before
linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#%% Ridge regression with regularization parameter: alpha

# best model fit ~ 20 - middle value

print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))