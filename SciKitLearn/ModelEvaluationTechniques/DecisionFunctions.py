# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:24:36 2023

@author: sueco
"""


#######################################################################
#%% Decision functions

# provide information about uncertainty associated with a particular prediciton

#Each classifer score value per test point indicates how confidenly the classifer predicts the positive class (large magnitude + values) or the negative class (lage - nevalues)

#Choosing a fixed decision threshold gives a classification rule
# By sweeping the decision threshold through the entire range of possible score values, we get a series of classification outcomes that form a curve.

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))

# show the decision_function scores for first 20 instances
y_score_list

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))

# show the probability of positive class for first 20 instances
y_proba_list

# You can change the Decision threshold for different tasks

##########################################