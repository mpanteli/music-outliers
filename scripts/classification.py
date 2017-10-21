# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:10:32 2016

@author: mariapanteli
"""
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import map_and_average
import util_feature_learning


FILENAMES = map_and_average.OUTPUT_FILES
TRANSFORM_LABELS = ['LDA', 'PCA', 'NMF', 'SSNMF', 'NA']
RANDOM_STATE = 12345


def load_data_from_pickle(filename):
    '''Loads dataset from pickle file.

    Parameters
    ----------
    filename : str
        Path to pickle file holding the dataset.

    Returns
    -------
    X : np.array, 2D
        The features as a matrix n_samples x f_features 
        (aggregated rhythmic, melodic, timbral, harmonic features).
    Y : np.array, 1D
        The class labels for each sample. 
    Yaudio : np.array, 1D
        The audio identifier for each sample. 
    '''
    X_list, Y, Yaudio = pickle.load(open(filename,'rb'))
    X = np.concatenate(X_list, axis=1)
    return X, Y, Yaudio


def feat_inds_from_pickle(filename):
    '''Returns the indices for rhythmic, melodic, timbral, 
    harmonic features for the aggregated feature vectors X. 
    
    Parameters
    ----------
    filename : str
        Path to pickle file holding the dataset.

    Returns
    -------
    feat_labels : list of str
        The label specifying the feature category (rhythm or melody etc.)
    feat_inds : list of np.array of int
        The indices of each feature category as numpy arrays. 
    '''
    X_list, Y, Yaudio = pickle.load(open(filename,'rb'))
    len_inds = np.array([X_list[0].shape[1], X_list[1].shape[1], 
                         X_list[2].shape[1], X_list[3].shape[1]])
    cum_sum = np.concatenate([[0], np.cumsum(len_inds)])
    feat_inds = [np.arange(cum_sum[i], cum_sum[i+1]) for i in range(len(X_list))]
    feat_labels = ['rhy', 'mel', 'mfc', 'chr']
    return feat_labels, feat_inds


def classify_for_filenames(file_list):
    '''Classification results for each projected feature space.

    Parameters
    ----------
    file_list : list of str
        A list of the file paths for each projected dataset.

    Returns
    -------
    df_results : pd.DataFrame
        The classification results for each projection. 
    '''
    df_results = pd.DataFrame()
    feat_learner = util_feature_learning.Transformer()
    for filename, transform_label in zip(file_list, TRANSFORM_LABELS):
        X, Y, Yaudio = load_data_from_pickle(filename)
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.6, 
                                            random_state=RANDOM_STATE, stratify=Y)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.5, 
                                            random_state=RANDOM_STATE, stratify=Y_val_test)
        df_result = classify_each_feature(X_train, Y_train, X_test, Y_test, filename, 
                                            transform_label=transform_label)
        df_results = pd.concat([df_results, df_result], axis=0, ignore_index=True)
    return df_results


def classify_each_feature(X_train, Y_train, X_test, Y_test, filename, transform_label=" "):
    '''Classification results for each feature category (rhythm, melody, timbre, harmony).

    Parameters
    ----------
    X_train : np.array, 2D
        The train data.
    Y_train : np.array, 1D
        The train class labels.
    X_test : np.array, 2D
        The test data.
    Y_test : np.array, 1D
        The test class labels.
    filename : str
        Path to file holding the projected dataset.
    transform_label : str
        The transform method ('PCA', 'LDA', 'NMF', 'SSNMF') used to project the dataset.

    Returns
    -------
    df_results : pd.DataFrame
        The classification results for each feature category. 
    '''
    n_dim = X_train.shape[1]
    feat_labels, feat_inds = feat_inds_from_pickle(filename)
    feat_learner = util_feature_learning.Transformer()
    # first the classification with all features together
    df_results = feat_learner.classify(X_train, Y_train, X_test, Y_test, 
                                        transform_label=transform_label)
    # then append for each feature separately
    for i in range(len(feat_inds)):
        df_result = feat_learner.classify(X_train[:, feat_inds[i]], Y_train, 
                                          X_test[:, feat_inds[i]], Y_test, transform_label=transform_label)
        df_results = pd.concat([df_results, df_result.iloc[:, 2]], axis=1, ignore_index=True)
    return df_results


def plot_CF(CF, labels=None, figurename=None):
    '''Plots the confusion matrix.

    Parameters
    ----------
    CF : np.array, 2D
        The confusion matrix
    labels : np.array, 1D
        The class labels of the confusion matrix.
    ffigurename : str
        The file name to output figure. If None, no output figure. 
    '''
    labels[labels=='United States of America'] = 'United States Amer.'
    plt.imshow(CF, cmap="Greys")
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    if figurename is not None:
        plt.savefig(figurename, bbox_inches='tight')


def confusion_matrix(X_train, Y_train, X_test, Y_test, classifier='LDA'):
    '''Classifies the data and estimates the confusion matrix.

    Parameters
    ----------
    X_train : np.array, 2D
        The train data.
    Y_train : np.array, 1D
        The train class labels.
    X_test : np.array, 2D
        The test data.
    Y_test : np.array, 1D
        The test class labels.
    classifier : str
        Specifies the classifier ('LDA', 'KNN', 'SVM', 'RF') to be used.

    Returns
    -------
    accuracy : np.float
        The classification f-score.
    CF : np.array, 2D
        The confusion matrix.
    labels : np.array, 1D
        The class labels of the confusion matrix.
    '''
    feat_learner = util_feature_learning.Transformer()
    if classifier=='LDA':
        model = feat_learner.modelLDA
    elif classifier=='KNN':
        model = feat_learner.modelKNN
    elif classifier=='SVM':
        model = feat_learner.modelSVM
    elif classifier=='RF':
        model = feat_learner.modelRF
    accuracy, predictions = feat_learner.classification_accuracy(X_train, Y_train, 
                        X_test, Y_test, model=model)
    labels = np.unique(Y_test)  # TODO: country labels in geographical proximity
    CF = metrics.confusion_matrix(Y_test, predictions, labels=labels)
    return accuracy, CF, labels 


def confusion_matrix_for_dataset(filename, classifier='LDA', output_data=False):
    '''Loads dataset, classifies, and returns confusion matrix.

    Parameters
    ----------
    filename : str
        Path to file holding the projected dataset.
    classifier : str
        Specifies the classifier ('LDA', 'KNN', 'SVM', 'RF') to be used.
    output_data : boolean
        Whether to export the confusion matrix as figure and csv. 

    Returns
    -------
    CF : np.array, 2D
        The confusion matrix.
    labels : np.array, 1D
        The class labels of the confusion matrix.
    '''
    X, Y, Yaudio = load_data_from_pickle(filename)
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.6, 
                                                        random_state=RANDOM_STATE, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.5, 
                                                        random_state=RANDOM_STATE, stratify=Y_val_test)
    accuracy, CF, labels = confusion_matrix(X_train, Y_train, X_test, Y_test, classifier=classifier)
    if output_data:
        np.savetxt('../data/CFlabels.csv', labels, fmt='%s')
        np.savetxt('../data/CF.csv', CF, fmt='%10.5f')
        plot_CF(CF, labels=labels, figurename='../data/conf_matrix.pdf')
    return CF, labels


if __name__ == '__main__':
    df_results = classify_for_filenames(file_list=FILENAMES)
    CF, labels = confusion_matrix_for_dataset(FILENAMES[0], output_data=False)

