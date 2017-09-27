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

import map_and_average
import util_feature_learning


FILENAMES = map_and_average.OUTPUT_FILES
TRANSFORM_LABELS = ['LDA', 'PCA', 'NMF', 'SSNMF', 'NA']
RANDOM_STATE = 12345

def load_data_from_pickle(filename):
    X_list, Y, Yaudio = pickle.load(open(filename,'rb'))
    X = np.concatenate(X_list, axis=1)
    return X, Y, Yaudio


def feat_inds_from_pickle(filename):
    X_list, Y, Yaudio = pickle.load(open(filename,'rb'))
    len_inds = np.array([X_list[0].shape[1], X_list[1].shape[1], 
                         X_list[2].shape[1], X_list[3].shape[1]])
    cum_sum = np.concatenate([[0], np.cumsum(len_inds)])
    feat_inds = [np.arange(cum_sum[i], cum_sum[i+1]) for i in range(len(X_list))]
    #feat_inds = [X_list[0].shape[1], X_list[1].shape[1], X_list[2].shape[1], X_list[3].shape[1]] 
    feat_labels = ['rhy', 'mel', 'mfc', 'chr']
    return feat_labels, feat_inds


def get_train_test_indices(audiolabs):
    trainset, valset, testset = map_and_average.load_train_val_test_sets()
    trainaudiolabels, testaudiolabels = trainset[2], testset[2]
    # train, test indices
    aa_train = np.unique(trainaudiolabels)
    aa_test = np.unique(testaudiolabels)
    traininds = np.array([i for i, item in enumerate(audiolabs) if item in aa_train])
    testinds = np.array([i for i, item in enumerate(audiolabs) if item in aa_test])
    return traininds, testinds


def get_train_test_sets(X, Y, traininds, testinds):
    X_train = X[traininds, :]
    Y_train = Y[traininds]
    X_test = X[testinds, :]
    Y_test = Y[testinds]
    return X_train, Y_train, X_test, Y_test


def classify_for_filenames(file_list=FILENAMES):
    df_results = pd.DataFrame()
    feat_learner = util_feature_learning.Transformer()
    #traininds, testinds = get_train_test_indices(Yaudio)
    for filename, transform_label in zip(file_list, TRANSFORM_LABELS):
        print filename
        X, Y, Yaudio = load_data_from_pickle(filename)
        #X_train, Y_train, X_test, Y_test = get_train_test_sets(X, Y, traininds, testinds)
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.6, random_state=RANDOM_STATE, stratify=Y)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.5, random_state=RANDOM_STATE, stratify=Y_val_test)
        #df_result = feat_learner.classify(X_train, Y_train, X_test, Y_test, transform_label=transform_label)
        #df_result_feat = classify_each_feature(X_train, Y_train, X_test, Y_test, filename, transform_label=transform_label)
        #df_result = pd.concat([df_result, df_result_feat], axis=1, ignore_index=True)
        #df_results = pd.concat([df_results, df_result], axis=0, ignore_index=True)
        df_result = classify_each_feature(X_train, Y_train, X_test, Y_test, filename, transform_label=transform_label)
        df_results = pd.concat([df_results, df_result], axis=0, ignore_index=True)
    return df_results


def classify_each_feature(X_train, Y_train, X_test, Y_test, filename, transform_label=" "):
    n_dim = X_train.shape[1]
    #feat_labels, feat_inds = map_and_average.get_feat_inds(n_dim=n_dim)
    feat_labels, feat_inds = feat_inds_from_pickle(filename)
    #df_results = pd.DataFrame()
    feat_learner = util_feature_learning.Transformer()
    # first the classification with all features together
    df_results = feat_learner.classify(X_train, Y_train, X_test, Y_test, transform_label=transform_label)
    # then append for each feature separately
    for i in range(len(feat_inds)):
        df_result = feat_learner.classify(X_train[:, feat_inds[i]], Y_train, 
                                          X_test[:, feat_inds[i]], Y_test, transform_label=transform_label)
        df_results = pd.concat([df_results, df_result.iloc[:, 2]], axis=1, ignore_index=True)
    return df_results


def plot_CF(CF, labels=None, figurename=None):
    labels[labels=='United States of America'] = 'United States Amer.'
    plt.imshow(CF, cmap="Greys")
    plt.xticks(range(len(labels)), labels, rotation='vertical', fontsize=4)
    plt.yticks(range(len(labels)), labels, fontsize=4)
    if figurename is not None:
        plt.savefig(figurename, bbox_inches='tight')


def confusion_matrix(X_train, Y_train, X_test, Y_test, saveCF=False, plots=False):
    feat_learner = util_feature_learning.Transformer()
    accuracy, predictions = feat_learner.classification_accuracy(X_train, Y_train, 
                        X_test, Y_test, model=feat_learner.modelLDA)
    labels = np.unique(Y_test)  # TODO: countries in geographical proximity
    CF = metrics.confusion_matrix(Y_test, predictions, labels=labels)
    if saveCF:
        np.savetxt('data/CFlabels.csv', labels, fmt='%s')
        np.savetxt('data/CF.csv', CF, fmt='%10.5f')
    if plots:
        plot_CF(CF, labels=labels, figurename='data/conf_matrix.pdf')
    return accuracy, CF 


def confusion_matrix_for_best_classification_result(df_results, output_data=False):
    max_i = np.argmax(df_results[:, 1])
    feat_learning_i = max_i % 4  # 4 classifiers for each feature learning method
    filename = FILENAMES[feat_learning_i]
    print filename
    X, Y, Yaudio = load_data_from_pickle(filename)
    #traininds, testinds = get_train_test_indices(Yaudio)
    #X_train, Y_train, X_test, Y_test = get_train_test_sets(X, Y, traininds, testinds)
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.6, random_state=RANDOM_STATE, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.5, random_state=RANDOM_STATE, stratify=Y_val_test)
    if output_data:
        _, CF = confusion_matrix(X_train, Y_train, X_test, Y_test, saveCF=True, plots=True)
    else:
        _, CF = confusion_matrix(X_train, Y_train, X_test, Y_test, saveCF=False, plots=False)
    return CF


if __name__ == '__main__':
    df_results = classify_for_filenames(file_list=FILENAMES)
    CF = confusion_matrix_for_best_classification_result(df_results, output_data=False)

