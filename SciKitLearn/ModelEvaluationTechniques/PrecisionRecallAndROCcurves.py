# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:25:39 2023

@author: sueco
"""

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


