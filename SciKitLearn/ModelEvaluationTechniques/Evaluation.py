# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 09:57:19 2023

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
#%% Confusion matricies

                    #Predicted Negative     Precicted Positive

# True Negative         TN                  FP

# True Positive         FN                  TP

#Breasks down classifer results by error type


# Binary (two class) confusion matrix
from sklearn.metrics import confusion_matrix

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
#%%  Precision recall curves

# top right corner is best place to be in the plot

# can see tradeoff between precision and recall as preceision increases, recall decreases

# Y: axis - just the recall rate
# points vary as we change decision treshold




from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
#pulling out the value for a threshold of zero
closest_zero = np.argmin(np.abs(thresholds))  
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
#plt.axes().set_aspect('equal')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)

plt.show()

###################################################
#%% ROC curves

# Used to evaluate performance of binary classifier

# X:axis: False Positive RAte
# Y-axis: True positive rate

# Top left corner:
        # the 'ideal point
        # False psotive rate = zero
            # True postive rate of one
# 'Steepness' of ROC curve is important:
    #Maximize the true positive rate
    #While minimizing the false positive rate

from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# LR is logistic regression
y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)

# computes the area under the curve to quantify quality
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
#put the area under curve (auc) value on plot
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
#plt.axes().set_aspect('equal')
plt.show()

# blue dashed line is results from just random selection
# 'bad' classifier will have results that lie close to blue dashed.  
# 'Reasonably good classifier will give ROC curve consistently better than random
# 'Excellent' classifier shown here

# Can quantify the quality of the model by looking at the area under the curve.  Random = about 0.5  Better = closer to 1

#%%  AUC  area under curve

# Advantages:
    # gives a single number for easy comparison
    # does not require specifying a decision threshold

# Drawbacks
    # As with other single-number metrics, AUC loses information (eg about tradeofss and the sahpe of the ROC curve)
    # This may be a factor to consider when wanting to compare the perforance of classifers with similar ROC curves


from matplotlib import cm

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
for g in [0.01, 0.1, 0.20, 1]:
    svm = SVC(gamma=g).fit(X_train, y_train)
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test)
    print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, 
                                                                    roc_auc_svm))
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="lower right", fontsize=11)
plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
#plt.axes().set_aspect('equal')

plt.show()




#####################################################################################
#%%  Evaluation measures for ****  multi-class classification  ***


# Can use similar metrics to binary cases
    # a collection of true vs predicted binary outcomes - one per class
    # confusion matrices are especially useful
    # classification reports for multiple metrics
    
# Different ways to average multi-class metrics
    # the number of instances for each class is importatnt to consider (i.e imballanced classes)
    
# Multi-label classification: each instance can have multiple labels (not convered here)

dataset = load_digits()
X, y = dataset.data, dataset.target
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

# linear kernel
svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)

svm_predicted_mc = svm.predict(X_test_mc)

confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)

df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,10)], columns = [i for i in range(0,10)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, 
                                                                       svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

#True classifications on the diagonal - miss classifications off the diagonal


# rbf kernel
svm = SVC(kernel = 'rbf').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(0,10)],
                  columns = [i for i in range(0,10)])

# shown as heat map
plt.figure(figsize = (5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM RBF Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, 
                                                                    svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label');

# Always look at confusion matrix for your classifier to get some insight into what kinds of errors it is making for each class

#%% Multi-class classification report

print(classification_report(y_test_mc, svm_predicted_mc))

#%% Micro vs macro averaged metrics

# Macro-average: Each class has equal weight
    # compute metric within each class
    # average resulting metrics across classes
    
# Micro-averagin: Each instance has equal weight
    # largest classes have most influence
    # aggregate outcomes across all classes
    # compute metric with aggregate outcomes
    


print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average = 'macro')))


# if the micro-average is much lower than the macro-average, then examine the larger classes for poor metric performance

# if the macro-average is much lower than the micro-average then examine the samller classes for poor metric performance
# averaged metrics
print('Micro-averaged f1 = {:.2f} (treat instances equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average = 'micro')))
print('Macro-averaged f1 = {:.2f} (treat classes equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average = 'macro')))


######################################################################
#%% Regression evaluation metrics

# typicall r2_score is enough
    # reminder: computes how well future instances will be predicted
    # best possible score is 1
    # constnt prediction score is 0
    
# Alternative metrics include:
    # mean_absolute_error
    # mean)squared_error        ()
    # median_absolute_error  (robust to outliers)
    
# Dummy regressor class
# dummy prediction model that uses a fixed rule can be useful. 

#%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

diabetes = datasets.load_diabetes()

X = diabetes.data[:, None, 6]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Real fit
lm = LinearRegression().fit(X_train, y_train)
#Dummy regressor
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

#predict read
y_predict = lm.predict(X_test)
#predict dummuy
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

# Other options
    # mean
    # median
    # quantile: predicts a user=provided quantile of the training taget values 
    # constant: preeicts a custom constant provided by user

print('Linear model, coefficients: ', lm.coef_)
print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test, 
                                                                     y_predict_dummy_mean)))
print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_predict, color='green', linewidth=2)
plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle = 'dashed', 
         linewidth=2, label = 'dummy')

plt.show()

##################################################################################
# Practical guide to controlled experiments on the the web

# Kohavi, R., Henne, R. M., & Sommerfield, D. (2007). Practical guide to controlled experiments on the web. Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD '07. doi:10.1145/1281192.1281295


# randomized experiments (single factor or factorial designs)
# A/B tests
# split tests
# Contro/Treatment tests
# parallel flights

# terminology: 
    #Overall Evaluation Criterion: Dependent variable, performance metric fitness function
    
    # Factor: controllable experimental variable that is thought to influence the OEC
    
    # Variant: Treatment vs control
    
    # Experimental Unit: the entity on which observations made.  Units independnet ideallyUse most commmon experimental unit on the web
    
    # A/A test - assume users to two groups but explose them to exactly the same experimence.  Can be used to collect data dn asses viable for power calculations
    # test the exerpimentation system





###########################################################################
#%%         Model selection using evaluation metrics


# fit data to both test and train data
    # leads to overfitting - but can serve as a sanity check to make sure pipeile is running correctly.
    
# multiple test-train splits - folds - cross validation
    # privides way to get out variance
    

    
    

#%%  Cross validation example

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

dataset = load_digits()
# again, making this a binary problem with 'digit 1' as positive class 
# and 'not 1' as negative class
X, y = dataset.data, dataset.target == 1
clf = SVC(kernel='linear', C=1)  # five folds default

# accuracy is the default scoring metric
print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))
# use AUC as scoring metric
print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))
# use recall as scoring metric
print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall'))

#%%  Grid search example

# used within each cross-validation fold to find optimal parameters for a given mentric

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
# gamma parameter sets with width of influence of the kernel
grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
#instantiate
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
# fit to data
grid_clf_acc.fit(X_train, y_train)

y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

#%%  Evaluation metrics supported for model selection

from sklearn.metrics import SCORERS

print(sorted(list(SCORERS.keys())))

#['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']


########################################################################
#%%          Two feature classification example using the digits dataset

# Optimizing a classifier using different evaluation metrics

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a two-feature input vector matching the example plot above
# We jitter the points (add a small amount of random noise) in case there are areas
# in feature space where many instances have the same features.
jitter_delta = 0.25
X_twovar_train = X_train[:,[20,59]]+ np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twovar_test  = X_test[:,[20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel = 'linear').fit(X_twovar_train, y_train)
grid_values = {'class_weight':['balanced', {1:2},{1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}
plt.figure(figsize=(9,6))
for i, eval_metric in enumerate(('precision','recall', 'f1','roc_auc')):
    grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric)
    grid_clf_custom.fit(X_twovar_train, y_train)
    print('Grid best parameter (max. {0}): {1}'
          .format(eval_metric, grid_clf_custom.best_params_))
    print('Grid best score ({0}): {1}'
          .format(eval_metric, grid_clf_custom.best_score_))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plot_class_regions_for_classifier_subplot(grid_clf_custom, X_twovar_test, y_test, None,
                                             None, None,  plt.subplot(2, 2, i+1))
    
    plt.title(eval_metric+'-oriented SVC')
plt.tight_layout()
plt.show()


#%%  Precision-recall curve for the default SVC classifier (with balanced class weights)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.svm import SVC

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create a two-feature input vector matching the example plot above
jitter_delta = 0.25
X_twovar_train = X_train[:,[20,59]]+ np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twovar_test  = X_test[:,[20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel='linear', class_weight='balanced').fit(X_twovar_train, y_train)

y_scores = clf.decision_function(X_twovar_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plot_class_regions_for_classifier(clf, X_twovar_test, y_test)
plt.title("SVC, class_weight = 'balanced', optimized for accuracy")
plt.show()

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.title ("Precision-recall curve: SVC, class_weight = 'balanced'")
plt.plot(precision, recall, label = 'Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
#plt.axes().set_aspect('equal')
plt.show()
print('At zero threshold, precision: {:.2f}, recall: {:.2f}'
      .format(closest_zero_p, closest_zero_r))

####################################################################
#%%
#  Training, validation and test framework for model selection and evaluation

# Use three data splits
#1 training set (model building)
# 2 Validation set (model selection)
# Test set (final evaluation)

