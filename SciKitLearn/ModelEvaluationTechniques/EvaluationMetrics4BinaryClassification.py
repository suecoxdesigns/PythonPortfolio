# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:21:50 2023

@author: sueco
"""
###################################################################
#%%  Evaluation metrics for binary classification


#other metrics of evaluation

    # accuracy: correct/total
    # classification error 1-Accuracy
    #Recall,  True Positive Rate (TPR): what fraction of all psoitive instances does the classifier correctly identify as positive   TP/ (TP +FN)
        # will be smaller by increasing TP or decreasing FN
    # Precision: what fraction of positve predictions are correct?
    # false positive rat (FPR)e: FP/TN+FP
        # specificity = 1-FPR
        
        # Precision and REcall are often tradeoffs - one goes up - the other goes down
      
    # F1 - score: combining precision and recall into a single number
    # 2* Precision*Recall/(Precission + Recall)  = 2*TP/(2* TP + FN+FP)
    
    # F:score: generalizaes F1-score for combining precision and recall into a single number
    
    #Fb = (1+B^2) *TP /  ((1+B^2)* TP + B*FN+FP) )
    # beta allows adustment of metric to control the emphasis on recall vs precision
    #Precision oriented users B=0.5
    # Recall-oriented uses: B=2
    
    
# How to decide what metric to apply?
# Is it more important to avoid false positives or false negatives?
    # min FP: use Precision
    # min FN: use Recall

###########################################################
#%% Confusion matricies

                    #Predicted Negative     Precicted Positive

# True Negative         TN                  FP

# True Positive         FN                  TP

#Breasks down classifer results by error type


# Binary (two class) confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

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

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('Most frequent class (dummy classifier)\n', confusion)

#%%
# produces random predictions w/ same class proportion as training set
dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_classprop_predicted = dummy_classprop.predict(X_test)
confusion = confusion_matrix(y_test, y_classprop_predicted)

print('Random class-proportional prediction (dummy classifier)\n', confusion)

#%%
# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print('Support vector machine classifier (linear kernel, C=1)\n', confusion)

#%%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

#%%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print('Decision tree classifier (max_depth = 2)\n', confusion)




#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 


# use (true labels , predicted labels) as inputs


print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))


# Combined report with all above metrics
from sklearn.metrics import classification_report

print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))

print('Random class-proportional (dummy)\n', 
      classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
print('SVM\n', 
      classification_report(y_test, svm_predicted, target_names = ['not 1', '1']))
print('Logistic regression\n', 
      classification_report(y_test, lr_predicted, target_names = ['not 1', '1']))
print('Decision tree\n', 
      classification_report(y_test, tree_predicted, target_names = ['not 1', '1']))

#######################################################################