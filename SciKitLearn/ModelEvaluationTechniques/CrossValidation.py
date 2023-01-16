# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:04:28 2023

@author: sueco
"""

###############################################################
#%% Cross validation
################################################################

# use Multiple train-test splits - each of which is used to generate a single model

# gives more stable a reliable estiamtes on how the classifier will work on new data by running multiple tries and averaging the results - makeing it less sensitive to random changes in test data

# "Cross validatation gives a more stable performance estimate of a given specific model so that it can be reliable compared with a different model
    # computer the mean evaluation metric across CV fols; an estimate not a guarantee
    # e.g. are SVM's better than Naive Bayes on this data for this metric?
    #Then train a final production classifier with the best model using ALL the data
    
    # 10 fold cross validation - averages across 10 different tries and averageing...
    
    #Typically cross-validation is NOT used to PRODUCE a model - but just used to estimate effectiveness
    
    # Bottom line: for reliable comparison of an evaluation metric across two different model types, use k-fold cross-validation NOT a single train-test split.
    
    ####  Evaluate and compare - the accuracy/score of different models with k-fold corss-validaiton
    
    ##### Tuning a single model (ege. find the optimal hyperparameters) uses a slightly different setup: train/validate/test split



#%% Example based k-NN classfier with fruit dataset (2 features) 

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# instantiate a classifier
clf = KNeighborsClassifier(n_neighbors = 5)

# define the x and y values
fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height','width','mass','color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple','mandarin','organge','lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X = X_fruits_2d.values
y = y_fruits_2d.values 


# call the cross_val_score function to get cv score
cv_scores = cross_val_score(clf, X, y, cv=10)

print('Cross-validation scores (10-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))

# by default cross validation does 5 fold...  change CV to number of 'folds' - here I've put cv=10


# We can use the variation in scores across folds to give us a measure of uncertatinly on the results of real data


### must be careful about stratified data - if all the data for one class are groups - the testing data may miss a category entirely... - so we can 'stratify' the data to equally distribute. skilearn does this automattically - distributes classes

# leave one out validation
# traiing set - all but one sample - TF need to do this n-1 times
# good for small sample sizes - help performance of model

# 


# ### A note on performing cross-validation for more advanced scenarios.

#In some cases (e.g. when feature values have very different ranges), we've seen the need to scale or normalize the training and test sets before use with a classifier. The proper way to do cross-validation when you need to scale the data is *not* to scale the entire dataset with a single transform, since this will indirectly leak information into the training data about the whole dataset, including the test data (see the lecture on data leakage later in the course).  Instead, scaling/normalizing must be computed and applied for each cross-validation fold separately.  To do this, the easiest way in scikit-learn is to use *pipelines*.  While these are beyond the scope of this course, further information is available in the scikit-learn documentation here:

#http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

#or the Pipeline section in the recommended textbook: Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido (O'Reilly Media).

#%%  Validation curve example

# Validation curves show sensitivity to changes in different parameters

#*************  Not used to tune a model  ********************

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

param_range = np.logspace(-3, 3, 4)

# define the x and y values
fruits = pd.read_table('assets/fruit_data_with_colors.txt')

feature_names_fruits = ['height','width','mass','color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple','mandarin','organge','lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X = X_fruits_2d.values
y = y_fruits_2d.values 

train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)

# returns two arrays : train_scores, test _scores
# number columns = number in range tested
# number of rows = number of folds tested (here 3)

# here SVC is the type of model_selection to use


# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()