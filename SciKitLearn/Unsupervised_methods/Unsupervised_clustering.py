# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:56:24 2023

@author: sueco
"""

####################################
#Datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Our sample fruits dataset
fruits = pd.read_table('assets/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1

##########################################################################
#%%  Clustering

#Finding a way to divide a dataset into groups ('clusters')

# Data points within the same cluster should be 'close' or 'similar' in some way

# Data points in different clusters should be 'far apart' or 'different'

# Clustering algorithms output a cluser membership index for each data pointN;
    # hard clustering: each datapoint belongs to exactly one cluster
    # soft (fuzzy) clustering: each datapoint is assigned a weight, score or probability of membership for each cluster

##################################################################

# K-means

# Most widtely used method of Clusering

#Algorithm:
    # Initialization: Pick the number of clusters k you want to find.  Then pick k random points to serve as an initial guess for the cluster centers
    
    # Step A: assign each data point to the nearest cluster center
    
    # Step B: update each cluster cetner by replacing it with the mean of all points assigned to that cluster (in step A)
    
    # Repeat steps A and B until the centers converge to a stable solution
    
    
    
    
    
   # Pros:
       # works well for simple clusters that are the same size, well sparated and globular shapes
    
    
# Cons: 
    # need to decided on number of clusters a priori
    # highly sensitive to starting guesses : so often run with 10 different initializations
    # Does not do well with iregular, complex clusters
    
    
    
# Categorical features: use k-medoids

    

# This example from the lecture video creates an artificial dataset with make_blobs, then applies k-means to find 3 clusters, and plots the points in each cluster identified by a corresponding color.

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

#%% other k-means example with scaling

#Example showing k-means used to find 4 clusters in the fruits dataset. Note that in general, 

# **********it's important to scale the individual features before applying k-means clustering.  ************

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler

fruits = pd.read_table('assets/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']].values
y_fruits = fruits[['fruit_label']] - 1

X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)  

kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X_fruits_normalized)

plot_labelled_scatter(X_fruits_normalized, kmeans.labels_, 
                      ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])


# K-means can predict the cluster of a new group - so can call the .fit and .predict functions seporately


#%%  Agglomerative clustering 

# cluster with an iterative bottom up approach

#Steps:

    # each data point is put into its own cluster of one item

    # A sequence of clusterings are done where the most similar two clusters at each stage are merged into a new cluster

    # Then this process is repreated until some stopping condition is met: often the condition is the number of clusters
    
    # set of algorith determines the most similar cluster by specifying one of several possible linkage critera:
        
        # 'ward': merges the two clusters that give the smallest increase in total variance within all the clusters
            # works best on most data sets: most often used
        
        # 'average' : merges the clusters that have the smallest average distance between points
            # works well if clusters have very different sizes
        
        #'complete': (maximum linkage) merges the two clusters that have the smallest maximum distance between points


from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

cls = AgglomerativeClustering(n_clusters = 4)
cls_assignment = cls.fit_predict(X)

# can't be used to predict new data - so use the fit_predit function

plot_labelled_scatter(X, cls_assignment, 
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])

#%% Creating a dendrogram (using scipy)

# Agglomerative clustering saves hierarchy of clustering
# can visualize this with a dendrogram - like a phylogenetic tree

# height of the new node: how far apart the two clusters were when they merged

# This dendrogram plot is based on the dataset created in the previous step with make_blobs, but for clarity, only 10 samples have been selected for this example, as plotted here:
    
X, y = make_blobs(random_state = 10, n_samples = 10)
plot_labelled_scatter(X, y, 
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])
print(X)

#And here's the dendrogram corresponding to agglomerative clustering of the 10 points above using Ward's method. The index 0..9 of the points corresponds to the index of the points in the X array above. For example, point 0 (5.69, -9.47) and point 9 (5.43, -9.76) are the closest two points and are clustered first.

from scipy.cluster.hierarchy import ward, dendrogram
plt.figure()
dendrogram(ward(X))
plt.show()

#%%  DBSCAN clustering

# "density-based spatial clustering of applications with noise."

#"clusters represent areas in the dataspace that are more dense with data points, while being separated by regions that are empty or at least much less densely populated"

# Advantage: 
    # don't need to specify number of clusters in advance
    # can work well with weird cluster shapes
    # can find outliers that shouldn't be assigned to any cluster 
    
# main parameters: 
    # min_samples
    # eps:

        
# All points that lie in a more dense region are called core samples

# for a given data point, if there are 'min_sample' of other datapoints that lie within a distance of 'eps', that given data pint is labeled as a core sample

# all core samples that are with a distance of eps units apart are put into the same cluster

#  noise: points that don't end up belonging to any cluser
# boundary points: points within a distance of EPS units from core points, bu tnot core points themselves

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state = 9, n_samples = 25)

dbscan = DBSCAN(eps = 2, min_samples = 2)

cls = dbscan.fit_predict(X)
print("Cluster membership values:\n{}".format(cls))

plot_labelled_scatter(X, cls + 1, 
        ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2'])

# can't use to predict groups of new data = so use fit_predict method

# if all memberships returned with label -1, eps or min_samples parameters need to be adjusted

# importatnt to use scaler preprocessing!!!

#make sure that when you use the cluster assignments from DBSCAN, you check for and handle the -1 noise value appropriately. Since this negative value might cause problems, for example, if the cluster assignment is used as an index into another array later on.


#%%  Clustering evaluation

# No ground truth: 
    
# consider task-based evaluation: Evluate clustering according to performance on a task that *does* have an objective basis for comparison

# example: the effectiveness of a cluster-based features for a supervised learning task.

# also hard to evaluate the meaning of the cluster distinctions