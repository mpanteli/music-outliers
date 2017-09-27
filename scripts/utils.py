# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:26:01 2017

@author: mariapanteli
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet


def get_outliers(X, chi2thr=0.975):
    robust_cov = MinCovDet().fit(X)
    MD = robust_cov.mahalanobis(X)
    chi2 = stats.chi2
    degrees_of_freedom = X.shape[1]
    threshold = chi2.ppf(chi2thr, degrees_of_freedom)
    y_pred = MD>threshold
    return threshold, y_pred, MD


def get_outliers_Mahal(X, chi2thr=0.975):
    n_samples = X.shape[0]
    inv_cov = np.linalg.inv(np.cov(X.T))
    col_mean = np.mean(X, axis=0)
    MD = np.zeros(n_samples, dtype='float')
    for i in range(n_samples):
        MD[i] = mahalanobis(X[i,:], col_mean, inv_cov)
    MD = MD ** 2
    degrees_of_freedom = X.shape[1]
    chi2 = stats.chi2
    threshold = chi2.ppf(chi2thr, degrees_of_freedom)
    y_pred = MD>threshold
    return threshold, y_pred, MD


def pca_data(X, min_variance=None):
    # rotate data to avoid singularity in Mahalanobis/covariance matrix
    model = PCA(whiten=True).fit(X)
    model.explained_variance_ratio_.sum()
    if min_variance is None:
        n_pc = X.shape[1]
    else:
        n_pc = np.where(model.explained_variance_ratio_.cumsum()>min_variance)[0][0]
    X_pca = PCA(n_components=n_pc, whiten=True).fit_transform(X)
    return X_pca, n_pc


def get_local_outliers_from_neighbors_dict(X, Y, w_dict, chi2thr=0.975, do_pca=False):
    uniq_labels = np.unique(Y)
    spatial_outliers = []
    for uniq_label in uniq_labels:
        countries_neighbors = w_dict[uniq_label]
        if len(countries_neighbors)==0:
            print uniq_label, " no neighbors found"
            continue
        inds_neighborhood = []
        for country in countries_neighbors:
            inds = np.where(Y==country)[0]
            inds_neighborhood.append(inds) # append neighboring countries
        if len(np.concatenate(inds_neighborhood))==0:
            print "no neighbors found"
            continue
        inds_neighborhood.append(np.where(Y==uniq_label)[0]) # append query country
        inds_neighborhood = np.concatenate(inds_neighborhood)
        if do_pca:
            XX, _ = pca_data(X[inds_neighborhood, :], min_variance=0.99)
        else:
            XX = X[inds_neighborhood, :]  # assume X is already in PCA
        print len(inds_neighborhood)
        if len(inds_neighborhood)<XX.shape[1]:
            print uniq_label, " neighborhood too small for number of features"
            continue
        threshold, y_pred, MD = get_outliers_Mahal(XX, chi2thr=chi2thr)
        counts = Counter(Y[inds_neighborhood[y_pred]])
        spatial_outliers.append([uniq_label, counts[uniq_label], threshold, y_pred, MD, counts, inds_neighborhood])
    return spatial_outliers


def best_n_clusters_silhouette(X, min_ncl=2, max_ncl=50, metric='euclidean'):
    ave_silh = []
    for i in range(min_ncl):
        ave_silh.append(np.nan) # for ncl=0, ncl=1 no clustering
    for ncl in range(min_ncl, max_ncl):
        print ncl
        cl_pred = KMeans(n_clusters=ncl, random_state=50).fit_predict(X)
        ave_silh.append(metrics.silhouette_score(X, cl_pred, metric=metric)) # silhouette avg
    ave_silh = np.array(ave_silh)
    bestncl = np.nanargmax(ave_silh)
    return bestncl, ave_silh


def get_cluster_freq_linear(X, Y, centroids):
    """ for each label in Y get the distribution of clusters by linear encoding
    """
    def encode_linear(X, centroids):
        """Linear encoding via the dot product
        """
        return np.dot(X, centroids.T)
    encoding = encode_linear(X, centroids)
    encoding_df = pd.DataFrame(data=encoding, index=Y) 
    encoding_df_sum = encoding_df.groupby(encoding_df.index).sum()
    cluster_freq = (encoding_df_sum - np.mean(encoding_df_sum)) / np.std(encoding_df_sum) 
    cluster_freq.index.name = 'labels'
    return cluster_freq


def get_cluster_predictions(X, n_clusters=10):
    cl_pred = KMeans(n_clusters=n_clusters, random_state=50).fit_predict(X)
    return cl_pred
