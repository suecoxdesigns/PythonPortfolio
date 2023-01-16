# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:01:22 2023

@author: sueco
"""



#################################################
#%% Support Vector Machines
###########################################

# Linear support Vector Machine

# linear models for classification
# still linear sum of weights and b 
# but take output and apply sign function to produce two possible values
#  If the taget value is above zero - one value - below zero another
# above or below a line decision boundary

# classifier Margin
# The maximum width the decision boundary area can be increased before hittind a data point

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 



# make a dataset
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)


this_C = 1
#Instantiate SVC and fit to training data
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
#can change 'kernel'
# C specifies the regularization as above

#define a C
# larger values of c: less regularization
    # fit the training data as well as posisble
    # each individual data point is importatnt to classify correctly
    # even if it means fitting a smaller margin decision boundary
# smaller values of C: more regularization
    # more tolerant of erros on individual data points
this_C = 1.0

#plot a few a version of this
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

title = 'Linear SVC, C = {:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

#Linear support Vector Machine: C parameter

from sklearn.svm import LinearSVC


X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
plt.tight_layout()

#%% Application to real data set

from sklearn.svm import LinearSVC

# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)


clf = LinearSVC().fit(X_train, y_train)


print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


################################################################
##########  Linear Model pros and cons ###########################

# Pros
    # simple and easy to train
    # fast prediction
    # scales well to a very large dataset
    # works well with space data
    # reasons for prediction are relatively easy to interpret
    
# Cons
    # for lower-dimensional data, other models may have superioe generalization performance
    # For classification, data my not be linearly separable

