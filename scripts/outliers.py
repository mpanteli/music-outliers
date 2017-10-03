# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:49:48 2016

@author: mariapanteli
"""

import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.cluster import KMeans

import utils
import utils_spatial


def country_outlier_df(counts, labels, normalize=False, out_file=None):
    if len(counts.keys()) < len(np.unique(labels)):
        for label in np.unique(labels):
            if not counts.has_key(label):
                counts.update({label:0})
    if normalize:
        norm_counts = normalize_outlier_counts(counts, Counter(labels))
        df = pd.DataFrame.from_dict(norm_counts, orient='index').reset_index()
    else:
        df = pd.DataFrame.from_dict(Counter(counts), orient='index').reset_index()
    df.rename(columns={'index':'Country', 0:'Outliers'}, inplace=True)
    # append number of recordings and number of outliers per country 
    df_n_country = pd.DataFrame.from_dict(Counter(labels), orient='index').reset_index()
    df_n_country.rename(columns={'index':'Country', 0:'N_Country'}, inplace=True)
    df_n_outliers = pd.DataFrame.from_dict(Counter(counts), orient='index').reset_index()
    df_n_outliers.rename(columns={'index':'Country', 0:'N_Outliers'}, inplace=True)
    df = pd.merge(df, df_n_country, on='Country', how='left')
    df = pd.merge(df, df_n_outliers, on='Country', how='left')
    if out_file is not None:
        df.to_csv(out_file, index=False)
    return df


def normalize_outlier_counts(outlier_counts, country_counts):
    '''Normalize a dictionary of outlier counts per country by 
        the total number of recordings per country
    '''
    norm_counts = {}
    for key in outlier_counts.keys():
        # dictionaries should have the same keys
        norm_counts[key] = float(outlier_counts[key]) / float(country_counts[key])
    return norm_counts


def get_outliers_df(X, Y, chi2thr=0.999, out_file=None):
    threshold, y_pred, MD = utils.get_outliers_Mahal(X, chi2thr=chi2thr)
    global_counts = Counter(Y[y_pred])
    df = country_outlier_df(global_counts, Y, normalize=True, out_file=out_file)
    return df, threshold, MD


def print_most_least_outliers_topN(df, N=10):
    sort_inds = df['Outliers'].argsort()  # ascending order
    #df_most = df[['Country', 'Outliers']].iloc[sort_inds[::-1][:N]]
    #df_least = df[['Country', 'Outliers']].iloc[sort_inds[:N]]
    df_most = df.iloc[sort_inds[::-1][:N]]
    df_least = df.iloc[sort_inds[:N]]
    print "most outliers " 
    print df_most
    print "least outliers " 
    print df_least
    

def load_metadata(Yaudio, metadata_file):
    df = pd.read_csv(metadata_file)
    df_audio = pd.DataFrame({'Audio':Yaudio})
    ddf = pd.merge(df_audio, df, on='Audio', suffixes=['', '_r']) # in the order of Yaudio
    return ddf


def print_clusters_metadata(df, cl_pred, out_file=None):
    def get_top_N_counts(labels, N=3):
        ulab, ucount = np.unique(labels, return_counts=True)
        inds = np.argsort(ucount)
        return zip(ulab[inds[-N:]],ucount[inds[-N:]])
    info = np.array([str(df['Country'].iloc[i]) for i in range(len(df))])
    styles_description = []
    uniq_cl = np.unique(cl_pred)
    for ccl in uniq_cl:
        inds = np.where(cl_pred==ccl)[0]
        styles_description.append(get_top_N_counts(info[inds], N=3))
    df_styles = pd.DataFrame(data=styles_description, index=uniq_cl)
    print df_styles.to_latex()
    if out_file is not None:
        df_styles.to_csv(out_file, index=False)


def load_data(pickle_file, metadata_file):
    X_list, Y, Yaudio = pickle.load(open(pickle_file,'rb'))
    ddf = load_metadata(Yaudio, metadata_file=metadata_file)
    w, data_countries = utils_spatial.get_neighbors_for_countries_in_dataset(Y)
    w_dict = utils_spatial.from_weights_to_dict(w, data_countries)
    ddf = utils_spatial.append_regions(ddf)
    return [X_list, Y, Yaudio], ddf, w_dict


def get_local_outliers_df(X, Y, w_dict, out_file=None):
    spatial_outliers = utils.get_local_outliers_from_neighbors_dict(X, Y, w_dict, chi2thr=0.999, do_pca=True)
    spatial_counts = Counter(dict([(ll[0],ll[1]) for ll in spatial_outliers]))
    df_local = country_outlier_df(spatial_counts, Y, normalize=True, out_file=out_file)
    return df_local


def get_country_clusters(X, bestncl=None, min_ncl=5, max_ncl=50):
    if bestncl is None:
        bestncl, ave_silh = utils.best_n_clusters_silhouette(X, min_ncl=min_ncl, max_ncl=max_ncl, metric="cosine")
    # get cluster predictions and metadata for each cluster
    cluster_model = KMeans(n_clusters=bestncl, random_state=50).fit(X)
    centroids = cluster_model.cluster_centers_
    cl_pred = cluster_model.predict(X)
    return centroids, cl_pred


if __name__ == '__main__':
    # load LDA-transformed frames
    dataset, ddf, w_dict = load_data('../data/lda_data_8.pickle', '../data/metadata.csv')
    X_list, Y, Yaudio = dataset
    X = np.concatenate(X_list, axis=1)

    # global outliers
    df_global, threshold, MD = get_outliers_df(X, Y, chi2thr=0.999)
    print_most_least_outliers_topN(df_global, N=10)

    # local outliers
    df_local = get_local_outliers_df(X, Y, w_dict)
    print_most_least_outliers_topN(df_local, N=10)

    # outliers for features
    feat = X_list
    feat_labels = ['rhy', 'mel', 'mfc', 'chr']
    tabs_feat = []
    for i in range(len(feat)):
        XX = feat[i]
        df_feat, threshold, MD = get_outliers_df(XX, Y, chi2thr=0.999)
        print_most_least_outliers_topN(df_feat, N=5)

    ## how many styles are there
    ##bestncl, ave_silh = utils.best_n_clusters_silhouette(X, min_ncl=5, max_ncl=50, metric="cosine")
    centroids, cl_pred = get_country_clusters(X, bestncl=10)
    ddf['Clusters'] = cl_pred
    print_clusters_metadata(ddf, cl_pred)

    # how similar are the cultures and which ones seem to be global outliers
    cluster_freq = utils.get_cluster_freq_linear(X, Y, centroids)
