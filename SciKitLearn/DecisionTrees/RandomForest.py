# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:29:21 2023

@author: sueco
"""

###################################################
#%%  Ensembles of Decision Trees
####################################################

# combines multiple learning models = more powerful than individual parts.

# By combining different individual models into an ensemble, we can average out their individual mistakes to reduce the risk of overfitting while maintaining strong prediction performance.


#%% Random forests

# creates lots of individual decision trees on a training set (often 10s or 100s)
# more stable better generalization
# Ensenble of trees should be diverse: introduct random variation into tree building

# randomness generated in two ways

    # 1) data selected randomly (bootstrap copies)
        # same number of samples as origional - but some rows potentially missing and some potentially duplicated
    # 2) Features chosen in each split test also randomly selected
    
    
# Key parameters
    #  n_estimator parameter:number of trees to build set
    
    # max_features: number of features to use
        # max_fetures = 1: 
            #forests with diverse, more complex trees
            # default works well in practice, but adjusting may lead to some further gains
            
        # max_features close to number of features
           # similar forests with simplier trees
    # max_depty: controls the depth of each tree (deault:None - splits until all leaves are pure)
    # n_jobs: How many cores to use in parallel during training
    
# choose a fixed setting for the random_state parameter if you need reproducible results
           
# Pros:
    # Widely used, excellent prediction performance on many problems
    # Doesn't require careful normalization of features or extensive parameter turning
    # Like decision trees, handles a mixture of feature types
    # EAstily paralleized across multiple CPUs
    
#Cons:
    # The resulting models are often diffiuclt for humans interpret
    # Like decision trees, random forests may not be a good choice for very high-deimensional tasks (e.g. text classifiers) compared to fact accuate linear models
    
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import  make_blobs
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

# make some data
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)


X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                   random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = RandomForestClassifier().fit(X_train, y_train)
title = 'Random Forest Classifier, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
                                         y_test, title, subaxes)

plt.show()

#%%  Random forest: Fruit dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import pandas as pd

# read in some data

# fruits dataset


fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']


X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']



X_train, X_test, y_train, y_test = train_test_split(X_fruits.values,  y_fruits.values, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

title = 'Random Forest, fruits dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = RandomForestClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()
plt.show()

#n_estimator = # of trees to build

clf = RandomForestClassifier(n_estimators = 10,
                            random_state=0).fit(X_train, y_train)

print('Random Forest, Fruit dataset, default settings')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#%% Random Forests on a real-world dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# 30 features in the data set
# didn't need to do scaling or other pre-processing
clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))