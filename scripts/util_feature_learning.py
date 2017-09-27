# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:14:40 2017

@author: mariapanteli
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer        
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition.pca import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from numpy.linalg import pinv

import nmftools


class Transformer:
    def __init__(self):
        self.pca_transformer = None
        self.lda_transformer = None
        self.nmf_transformer = None
        self.ssnmf_H = None
        self.modelKNN = None
        self.modelLDA = None
        self.modelSVM = None
        self.modelRF = None
        
        
    def ssnmf_fit(self, data, labels, npc=None):
        binarizer = LabelBinarizer()
        F_class = binarizer.fit_transform(labels)
        F, G, W, H, cost = nmftools.ssnmf(data, R=npc, F=F_class, n_iter=200)
        ssWH = np.dot(F, G) + np.dot(W, H)
        rec_err = np.linalg.norm(data - ssWH)
        return G, W, H, rec_err
    
    
    def fit_lda_data(self, X_train, Y_train, n_components=None, pca_only=False):
        X_train = scale(X_train, axis=0)
        # then pca
        print "training with PCA transform..."
        self.pca_transformer = PCA(n_components=n_components).fit(X_train)
        print "variance explained " + str(np.sum(self.pca_transformer.explained_variance_ratio_))
        if pca_only:
            # return pca transformer only
            return
        # then lda
        print "training with LDA transform..."
        self.lda_transformer = LDA(n_components=n_components).fit(X_train, Y_train)
        print "variance explained " + str(np.sum(self.lda_transformer.explained_variance_ratio_))
        
    
    def transform_lda_data(self, X_test):
        X_test = scale(X_test, axis=0)
        print "transform test data..."
        pca_testdata = self.pca_transformer.transform(X_test)
        lda_testdata = self.lda_transformer.transform(X_test)
        transformed_data = {'none': X_test, 'pca': pca_testdata, 
                                            'lda': lda_testdata,
                                            'nmf': [],
                                            'ssnmf': []}
        return transformed_data
    
    
    def fit_data(self, X_train, Y_train, n_components=None, pca_only=False):
        if n_components is None:
            n_components = X_train.shape[1]
        X_train = scale(X_train, axis=0)
        # then pca
        print "training with PCA transform..."
        self.pca_transformer = PCA(n_components=n_components).fit(X_train)
        print "variance explained " + str(np.sum(self.pca_transformer.explained_variance_ratio_))
        if pca_only:
            # return pca transformer only
            return
        # then lda
        print "training with LDA transform..."
        self.lda_transformer = LDA(n_components=n_components).fit(X_train, Y_train)
        print "variance explained " + str(np.sum(self.lda_transformer.explained_variance_ratio_))
        # then nmf
        print "training with NMF transform..."
        norm_traindata = normalize(X_train - np.min(X_train))
        self.nmf_transformer = NMF(n_components=n_components).fit(norm_traindata)
        print "reconstruction error " + str(np.sum(self.nmf_transformer.reconstruction_err_))
        # then ssnmf
        print "training with SSNMF transform..."
        G, W, self.ssnmf_H, rec_err = self.ssnmf_fit(norm_traindata, Y_train, npc=n_components)
        print "reconstruction error " + str(rec_err)
    
    
    def transform_data(self, X_test):
        X_test = scale(X_test, axis=0)
        print "transform test data..."
        pca_testdata = self.pca_transformer.transform(X_test)
        lda_testdata = self.lda_transformer.transform(X_test)
        norm_testdata = normalize(X_test - np.min(X_test))
        nmf_testdata = self.nmf_transformer.transform(norm_testdata)
        ssnmf_testdata = np.dot(norm_testdata, pinv(self.ssnmf_H))
        transformed_data = {'none': X_test, 'pca': pca_testdata, 
                                            'lda': lda_testdata, 
                                            'nmf': nmf_testdata, 
                                            'ssnmf': ssnmf_testdata}
        return transformed_data
    
    
    def classification_accuracy(self, X_train, Y_train, X_test, Y_test, model=None):
        if model is None:
            model = LDA()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        accuracy = metrics.f1_score(Y_test, predictions, average='weighted')  # for imbalanced classes
        return accuracy, predictions
        

    def classify(self, X_train, Y_train, X_test, Y_test, transform_label=" "):
        self.modelKNN = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        self.modelLDA = LDA()
        self.modelSVM = svm.SVC(kernel='rbf', gamma=0.1)
        self.modelRF = RandomForestClassifier()
        model_labels = ['KNN', 'LDA', 'SVM', 'RF']
        models = [self.modelKNN, self.modelLDA, self.modelSVM, self.modelRF]        
        df_results = pd.DataFrame()        
        for model, model_label in zip(models, model_labels):
            acc, _ = self.classification_accuracy(X_train, Y_train, X_test, Y_test, model=model)
            print model_label + " " + transform_label + " " + str(acc)
            df_results = df_results.append(pd.DataFrame([[transform_label, model_label, acc]]))
        return df_results


if __name__ == '__main__':
    Transformer()