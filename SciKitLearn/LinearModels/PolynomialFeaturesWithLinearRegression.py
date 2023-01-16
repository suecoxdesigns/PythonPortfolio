# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:54:03 2023

@author: sueco
"""

        
################################################################        
#%% Polynomial faetures with linear regression
#########################################################

# polynomial features with linear regression

# start with two features x = x0,x1
# make 3 more by combining them  x' = (x0,x1,x0^2, x0x1,x1^2)

# Can still use linear regression because y is still a linear combination

# The degreee of the polynomical specifies how many variables participate at a time in each new feature (above, deg = 2)

# Why do this?
# explore a wider space of functions to fit to the data
# but still using the same leas squares cost function
# negatives: have potential for overfitting - so often use regularized method

# doesn't have to only be polynomial - could be any non-linear fuction
#  ***************  non-linear basis functions  ******************



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


# make some data 
# synthetic dataset for more complex regression
from sklearn.datasets import make_friedman1
plt.figure()
plt.title('Complex regression problem with one input variable')
X_F1, y_F1 = make_friedman1(n_samples = 100,
                           n_features = 7, random_state=0)
plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
plt.show()


#First split up the data
X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1,
                                                   random_state = 0)
#Lets first do this with just straight up linear regression
# instantiate and fit a linear regression model
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))

# R-squared score (training): 0.969
#R-squared score (test): 0.805

print('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n')

#Instantiate a PolyFeatures prepocessing function
poly = PolynomialFeatures(degree=2)

# fit it to the x train data and transform the data
X_F1_poly = poly.fit_transform(X_F1)

#Use the transformed data 
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
     .format(linreg.score(X_test, y_test)))

#This doesn't help much

# (poly deg 2) linear model intercept (b): -3.206
#(poly deg 2) R-squared score (training): 0.969
#(poly deg 2) R-squared score (test): 0.805

####   Use Ridge with regularization

print('\nAddition of many polynomial features often leads to\n\
overfitting, so we often use polynomial features in combination\n\
with regression that has a regularization penalty, like ridge\n\
regression.\n')

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                   random_state = 0)
linreg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))

#This decreases over fitting and improves generalization

#(poly deg 2 + ridge) linear model intercept (b): 5.418
#(poly deg 2 + ridge) R-squared score (training): 0.826
#(poly deg 2 + ridge) R-squared score (test): 0.825

# and we can see which is the best fit
#[ 0.    2.23  4.73 -3.15  3.86  1.61 -0.77 -0.15 -1.75  1.6   1.37  2.52
#  2.72  0.49 -1.94 -1.63  1.51  0.89  0.26  2.05 -1.93  3.62 -0.72  0.63
# -3.16  1.29  3.55  1.73  0.94 -0.51  1.7  -1.98  1.81 -0.22  2.88 -0.89]