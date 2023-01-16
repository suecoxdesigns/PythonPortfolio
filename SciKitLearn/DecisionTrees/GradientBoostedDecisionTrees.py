# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:23:01 2023

@author: sueco
"""

#%%  Gradient-boosted decision trees

# use ensenble of multiple trees
# build a  series of trees, where each tree is trained so that it attempts to correct the mistakes of the previous tree in the series.

# parameters to vary
    # learning rate: controls how hard each new tree tries to correct remaining mistakes from previous round
        # high learning rate: more complex trees
        # low learning rate: simpler trees
        
#Decision boundaries have box like shape characteristic of decision trees - but more complex


#Pros:
    # often best off-the-self accuracy on many problems
    # Using model for prediction requires only modest memory and is fast
    # Doesnt' require normalization of features to perform well
    # like decision trees, handles a mixure of feature types
    
    
# cons:
    # LIke random forests, models are often difficult for humans to interpret
    # Requires careful tuning of the learning rate and other parameters
    # Training can require significant computation
    # like decision tress, not recommended for text classification and other problems with very high dimensional sparse features, for accuracyand computational cost reasons.
    
#Key parameters

#n_estimators: sets the # of small decision tress to use 
# learning_rate: controls emphasis on fixing errors from previous itteration

#The above two are typically tuned together
# n_estimators is adjusted first, to best exploit memory and CPU during training, then other parameters

#max_depth is typically set to a small value for most applications

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import  make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])


# make some data

# more difficult synthetic dataset for classification (binary)
# with classes that are not linearly separable
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

# visualize the data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,  marker= 'o', s=50, cmap=cmap_bold)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))


clf = GradientBoostingClassifier().fit(X_train, y_train)

# default: 
    # learningr_rate = 0.1
    # n_estimators = 100
    # max_depth = 3
    
    
title = 'GBDT, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
                                         y_test, title, subaxes)

plt.show()

#%% Gradient bosted decision trees on the fruit dataset

fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']


X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits.values,
                                                   y_fruits.values,
                                                   random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = GradientBoostingClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()
plt.show()
clf = GradientBoostingClassifier().fit(X_train, y_train)

print('GBDT, Fruit dataset, default settings')
print('Accuracy of GBDT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#%%  Gradient-boosted decision trees on a real-world dataset

from sklearn.ensemble import GradientBoostingClassifier

# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

#using default settings
clf = GradientBoostingClassifier(random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')
print('Accuracy of GBDT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))

#probably overfitting: to solve:
    # decrease learning rate
    # decrease max depth


clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')
print('Accuracy of GBDT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# amoung the best off-the-shelf supervised learning methods available