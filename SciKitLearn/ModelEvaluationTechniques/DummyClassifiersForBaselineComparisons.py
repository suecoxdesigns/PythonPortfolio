# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:17:20 2023

@author: sueco
"""



#  Represent 
# Train
# Evaluate
# Refine


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name,class_count)
    
#%%
 
# Creating a dataset with imbalanced binary classes:  
# Negative class (0) is 'not digit 1' 
# Positive class (1) is 'digit 1'
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30]) 

np.bincount(y_binary_imbalanced)    

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)

####################################################################
#%%  Dummy classifiers

#DummyClassifier is a classifier that makes predictions using simple rules, which can be useful as a baseline for comparison against actual classifiers, especially with imbalanced classes.

#Optional strategires

# 'Most_frequent' : predicts the most frequent label in the training set.
# 'stratified': random predictions based on traning set class distribution
# 'uniform',: generates predictions uniformly at random
# 'constant': always predicts a contant lavel provided by the user

from sklearn.dummy import DummyClassifier

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)

y_dummy_predictions

dummy_majority.score(X_test, y_test)

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)

#%%  Dummy REgressors

# serve same role as dummy classifiers but for regression tasks
#sanity check for regressor learners

#stragety prameter options:
    # mean: predicts mean of the traning targets
    # median: predicts the median of the training targets
    # quantile: predicts a user-provoded quantile of the training targets
    # constant: predicts a constant user-provided value


###########################################################