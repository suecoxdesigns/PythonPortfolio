# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:07:24 2023

@author: sueco
"""


##################################################
#%% Multi-class classification with linear models
#############################################################

#scilearn turns multi-class problems into a group of n individual classification problems.

# new data compared against each classifier and one with the highest scroe is returned

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height','width','mass','color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple','mandarin','organge','lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

# Break up the data set into a test and train set
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)


# Linear SVC with M classes generates M one vs rest classifiers

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

#Coeffcients [0] define apple vs the rest of the fruit
# makes a linear decision boundary for 'is apple ' or 'not applie'

#Coefficients show us the weights of the parameters for each feature

#%%  Multi-class results on the fruit data set

plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
           c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()
