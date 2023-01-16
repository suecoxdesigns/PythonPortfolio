# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:49:40 2023

@author: sueco
"""

#%% Lasso regression

# adds a regularization penalty term
# but abs(w) rather than w^2 for Ridge

# results in parmater weights to zero for least influential variables.  This is a ***sparse** solution : a kind of feature selection

#  The parameter alpha controls the amount of regularization (default = 1)

# The prediciton formulat is the same as ordinary least-squares

#****  When to use ridge vs lasso

# Use Ridge: many small/medium sized effects
# Use Lasso: few variables with medium/large effects

from adspy_shared_utilities import load_crime_dataset
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np



scaler = MinMaxScaler()
(X_crime, y_crime) = load_crime_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')

# linlasso.coef_ is the fit coefficients for all the features in the X_crime dataset

#list(X_crime) returns all the names of the features

# so list(zip(list(X_crime), linlasso.coef_)  returns a list of tuples where the first element is the name of the feature and the second element is the fit w.

#sorted( x, key = lambda e: -abs(e[1])), sorts on the function returned by the lambda - here the - absolute value of the coefficient

for e in sorted (list(zip(list(X_crime), linlasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))
        
#%% Lasso regression with regularization parameter: alpha

print('Lasso regression: effect of alpha regularization\n\
parameter on number of features kept in final model\n')

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
   
        
        