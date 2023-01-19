# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:55:36 2023

@author: sueco
"""


###########################
#%%  Transformations
######################

# Density Estimates: when you have a set of measruments scattered thoughout an area and you want to create what you can think of as a smooth version over the whole area that gives a general estamte of how likely it would be to observe a particular measrument in some area of that space.
# calculates a continuous probability density over the feature space given a set of discrete samples

# Kernel density is common technique here
    # particularly popular in creating heat maps with geospatial data

####################################
#Datasets
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Our sample fruits dataset
fruits = pd.read_table('assets/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1

#%%  Dimensionality Reduction and Manifold learning

# Finds an approximate version of your dataset using fewer features
# used for exploring and visualizing a dataset to understand grouping relationships
# Often visualized using a 2-dimensional scatterplot
# Also used for compression, finding features for supervised learning

#%%
# Principle Components Analysis (PCA)

# take your cloud of origional datapoints and finds a rotation of it so the dimeions are statisticlly uncorrelated.  PCA then typically drops all but the most informative initial dimensions that catpure most of the variation in the origional dataset


# Using PCA to find the first two principle components of the breat cancer dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# ************  Need to transform data set first!!  **************
# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

pca = PCA(n_components = 2).fit(X_normalized)

X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

#%%  Plotting the PCA-transfomred version of the break cancer dataset

from adspy_shared_utilities import plot_labelled_scatter
plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Breast Cancer Dataset PCA (n_components = 2)');

#%%  Plotting the magnitude of each feature value for the first two principle components

fig = plt.figure(figsize=(8, 5))
plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
feature_names = list(cancer.feature_names)

plt.gca().set_xticks(np.arange(-.5, len(feature_names)));
plt.gca().set_yticks(np.arange(0.5, 2));
plt.gca().set_xticklabels(feature_names+[""], rotation=90, ha='left', fontsize=12);
plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12); 

plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0,
                                              pca.components_.max()], pad=0.65);

# can be used to find features that can be used later in a supervised learning model
# here linear model would work well

#%% PCA on the fruit dataset (for comparison)

#%load_ext autoreload
#%autoreload 2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  

pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)

from adspy_shared_utilities import plot_labelled_scatter
plot_labelled_scatter(X_pca, y_fruits, ['apple','mandarin','orange','lemon'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Fruits Dataset PCA (n_components = 2)');


#%%  Manifold learning methods

# good at finding low dimensional structure in high dimensional data and are very useful for visualizations

# one example: where all points lie on a two-dimensional sheet (manifold) with an interesting shape.  

# visualize a high dimensional dataset and project it onto a lower dimensional space in cmost cases a two dimensional page - that in a way that preserves information about how points are close to eachother


#%% multidimensional scaling (MDS) on the fruit dataset

from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

# each feature should be centered (zero mean) and with unit variance
X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  

mds = MDS(n_components = 2)

X_fruits_mds = mds.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First MDS feature')
plt.ylabel('Second MDS feature')
plt.title('Fruit sample dataset MDS');

#%% MDS on the breat cancer dataset

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

mds = MDS(n_components = 2)

X_mds = mds.fit_transform(X_normalized)

from adspy_shared_utilities import plot_labelled_scatter
plot_labelled_scatter(X_mds, y_cancer, ['malignant', 'benign'])

plt.xlabel('First MDS dimension')
plt.ylabel('Second MDS dimension')
plt.title('Breast Cancer Dataset MDS (n_components = 2)');

#%% t-SNE on the fruit dataset

#t-SNE finds a two-dimensional representation of your data, such that the distances between points in the 2D scatterplot match as closely as possible the distances between the same points in the original high dimensional dataset. In particular, t-SNE gives much more weight to preserving information about distances between points that are neighbors

# t-SNE doesn't work well on the fruit data set - t-SNE tends to work better on datasets that have more well-defined local structure; in other words, more clearly defined patterns of neighbors 

from sklearn.manifold import TSNE

tsne = TSNE(random_state = 0)

X_tsne = tsne.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_tsne, y_fruits, 
    ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Fruits dataset t-SNE');

#%% t-SNE on the breast cancer dataset

tsne = TSNE(random_state = 0)

X_tsne = tsne.fit_transform(X_normalized)

plot_labelled_scatter(X_tsne, y_cancer, 
    ['malignant', 'benign'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Breast cancer dataset t-SNE');


#%%  How to use t-SNE effectively

#The t-SNE technique really is useful—but only if you know how to interpret it

#'Perplexity' :how to balance attention between local and global aspects of your data. The parameter is, in a sense, a guess about the number of close neighbors each point has.  Getting the most from t-SNE may mean analyzing multiple plots with different perplexities.

# between 5 and 50
# the perplexity should be smaller than the number of points

#Number of steps:  importatnt to go to stability
# If you see a t-SNE plot with strange “pinched” shapes, chances are the process was stopped too early

# Cluster size means nothing
# distances between clusters might not mean anything either
# Random noise does not always look random

#There’s a reason that t-SNE has become so popular: it’s incredibly flexible, and can often find structure where other dimensionality-reduction algorithms cannot. Unfortunately, that very flexibility makes it tricky to interpret. Out of sight from the user, the algorithm makes all sorts of adjustments that tidy up its visualizations. Don’t let the hidden “magic” scare you away from the whole technique, though. The good news is that by studying how t-SNE behaves in simple cases, it’s possible to develop an intuition for what’s going on.


##########################################################################