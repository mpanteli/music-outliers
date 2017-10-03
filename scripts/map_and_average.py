# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 02:44:07 2017

@author: mariapanteli
"""

import numpy as np
import pickle

import util_feature_learning
    
WIN_SIZE = 8
INPUT_FILES = ['../data/train_data_'+str(WIN_SIZE)+'.pickle', 
               '../data/val_data_'+str(WIN_SIZE)+'.pickle', 
               '../data/test_data_'+str(WIN_SIZE)+'.pickle']
OUTPUT_FILES = ['../data/lda_data_'+str(WIN_SIZE)+'.pickle', 
                '../data/pca_data_'+str(WIN_SIZE)+'.pickle', 
                '../data/nmf_data_'+str(WIN_SIZE)+'.pickle', 
                '../data/ssnmf_data_'+str(WIN_SIZE)+'.pickle', 
                '../data/na_data_'+str(WIN_SIZE)+'.pickle']


def remove_inds(features, labels, audiolabels):
    '''remove instances with unknown country
    '''
    remove_inds1 = np.where(labels=='unknown')[0]
    remove_inds2 = np.where(labels=='Unidentified')[0]
    keep_inds = np.array(list(set(range(len(labels))) - (set(remove_inds1) | set(remove_inds2))))
    features = features[keep_inds, :]
    labels = labels[keep_inds]
    audiolabels = audiolabels[keep_inds]
    return features, labels, audiolabels


def averageframes(features, audiolabels, classlabels):
    '''average frame-based features for each recording
    '''
    u, ind = np.unique(audiolabels, return_index=True)
    uniqsorted = u[np.argsort(ind)]
    newfeatures = []
    newclasslabels = []
    newaudiolabels = []
    for aulabel in uniqsorted:
        inds = np.where(audiolabels == aulabel)[0]
        newfeatures.append(np.mean(features[inds, :], axis=0))
        newclasslabels.append(classlabels[inds[0]])
        newaudiolabels.append(aulabel)
    newfeatures = np.array(newfeatures)
    newaudiolabels = np.array(newaudiolabels)
    newclasslabels = np.array(newclasslabels)
    return newfeatures, newaudiolabels, newclasslabels


def load_data_from_pickle(pickle_file=None):
    '''load frame based features and labels from pickle file
    '''
    with open(pickle_file,'rb') as f:
        data, labels, audiolabels = pickle.load(f)
    # remove 'unknown' and 'unidentified' country
    data, labels, audiolabels = remove_inds(data, labels, audiolabels)
    # avoid nan which gives error in feature learning 
    data[np.isnan(data)] = 0
    return data, labels, audiolabels


def load_train_val_test_sets():
    '''load train, val, test sets
    '''
    trainset = load_data_from_pickle(INPUT_FILES[0])
    valset = load_data_from_pickle(INPUT_FILES[1])
    testset = load_data_from_pickle(INPUT_FILES[2])
    return trainset, valset, testset


def limit_to_n_seconds(dataset, n_sec=30.0, win_sec=8.0):
    X, Y, Yaudio = dataset
    uniq_audio, uniq_counts = np.unique(Yaudio, return_counts=True)
    frame_sr = 2.0
    max_n_frames = np.int(np.floor((n_sec - win_sec) * frame_sr))
    X_new, Y_new, Yaudio_new = [], [], []
    for audio in uniq_audio:
        idx = np.where(Yaudio==audio)[0]
        if len(idx) > max_n_frames:
            idx = idx[:max_n_frames]
        X_new.append(X[idx, :])
        Y_new.append(Y[idx])
        Yaudio_new.append(Yaudio[idx])
    return [np.concatenate(X_new), np.concatenate(Y_new), np.concatenate(Yaudio_new)]


def get_feat_inds(n_dim=840):
    '''assume frame with 840 features and return indices for each feature
    '''
    if n_dim == 840:
        rhy_inds = np.arange(400)
        mel_inds = np.arange(400, 640)
        mfc_inds = np.arange(640, 720)
        chr_inds = np.arange(720, 840)
    elif n_dim == 640:
        rhy_inds = np.arange(200)
        mel_inds = np.arange(200, 440)
        mfc_inds = np.arange(440, 520)
        chr_inds = np.arange(520, 640)
    elif n_dim == 460:
        rhy_inds = np.arange(200)
        mel_inds = np.arange(200, 260)
        mfc_inds = np.arange(260, 340)
        chr_inds = np.arange(340, 460)
    elif n_dim == 660:
        rhy_inds = np.arange(400)
        mel_inds = np.arange(400, 460)
        mfc_inds = np.arange(460, 540)
        chr_inds = np.arange(540, 660)
    feat_inds = [rhy_inds, mel_inds, mfc_inds, chr_inds] 
    feat_labels = ['rhy', 'mel', 'mfc', 'chr']
    return feat_labels, feat_inds


def map_and_average_frames(dataset=None, n_components=None, min_variance=None):
    if dataset is None:
        trainset, valset, testset = load_train_val_test_sets()
    else:
        trainset, valset, testset = dataset
    traindata, trainlabels, trainaudiolabels = trainset
    valdata, vallabels, valaudiolabels = valset
    testdata, testlabels, testaudiolabels = testset
    print traindata.shape, valdata.shape, testdata.shape
    labels = np.concatenate((trainlabels, vallabels, testlabels)).ravel()
    audiolabels = np.concatenate((trainaudiolabels, valaudiolabels, testaudiolabels)).ravel()
    
    feat_labels, feat_inds = get_feat_inds(n_dim=traindata.shape[1])
    ldadata_list = []
    pcadata_list = []
    nmfdata_list = []
    ssnmfdata_list = []
    data_list = []
    for i in range(len(feat_inds)):
        print "mapping " + feat_labels[i]
        inds = feat_inds[i]
        ssm_feat = util_feature_learning.Transformer()
        if min_variance is not None:
            ssm_feat.fit_data(traindata[:, inds], trainlabels, n_components=len(inds), pca_only=True)
            n_components = np.where(ssm_feat.pca_transformer.explained_variance_ratio_.cumsum()>min_variance)[0][0]+1
            print n_components, len(inds)
            ssm_feat.fit_data(traindata[:, inds], trainlabels, n_components=n_components)
        elif n_components is not None:
            ssm_feat.fit_data(traindata[:, inds], trainlabels, n_components=n_components)
        else:
            ssm_feat.fit_data(traindata[:, inds], trainlabels, n_components=len(inds))
        all_data = np.concatenate((traindata[:, inds], valdata[:, inds], testdata[:, inds]), axis=0)
        transformed_data_dict = ssm_feat.transform_data(all_data)
        for key in transformed_data_dict.keys():
            average_data, audiolabs, classlabs = averageframes(transformed_data_dict[key], audiolabels, labels)
            transformed_data_dict[key] = average_data
        data_list.append(transformed_data_dict['none'])
        pcadata_list.append(transformed_data_dict['pca'])
        ldadata_list.append(transformed_data_dict['lda'])
        nmfdata_list.append(transformed_data_dict['nmf'])
        ssnmfdata_list.append(transformed_data_dict['ssnmf'])
    return data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs


def lda_map_and_average_frames(dataset=None, n_components=None, min_variance=None):
    if dataset is None:
        trainset, valset, testset = load_train_val_test_sets()
    else:
        trainset, valset, testset = dataset
    traindata, trainlabels, trainaudiolabels = trainset
    valdata, vallabels, valaudiolabels = valset
    testdata, testlabels, testaudiolabels = testset
    print traindata.shape, valdata.shape, testdata.shape
    labels = np.concatenate((trainlabels, vallabels, testlabels)).ravel()
    audiolabels = np.concatenate((trainaudiolabels, valaudiolabels, testaudiolabels)).ravel()
    
    feat_labels, feat_inds = get_feat_inds(n_dim=traindata.shape[1])
    ldadata_list = []
    pcadata_list = []
    nmfdata_list = []
    ssnmfdata_list = []
    data_list = []
    for i in range(len(feat_inds)):
        print "mapping " + feat_labels[i]
        inds = feat_inds[i]
        ssm_feat = util_feature_learning.Transformer()
        if min_variance is not None:
            ssm_feat.fit_lda_data(traindata[:, inds], trainlabels, n_components=len(inds), pca_only=True)
            n_components = np.where(ssm_feat.pca_transformer.explained_variance_ratio_.cumsum()>min_variance)[0][0]+1
            print n_components, len(inds)
            ssm_feat.fit_lda_data(traindata[:, inds], trainlabels, n_components=n_components)
        elif n_components is not None:
            ssm_feat.fit_lda_data(traindata[:, inds], trainlabels, n_components=n_components)
        else:
            ssm_feat.fit_lda_data(traindata[:, inds], trainlabels, n_components=len(inds))
        all_data = np.concatenate((traindata[:, inds], valdata[:, inds], testdata[:, inds]), axis=0)
        transformed_data_dict = ssm_feat.transform_lda_data(all_data)
        for key in transformed_data_dict.keys():
            if len(transformed_data_dict[key])==0:
                continue
            average_data, audiolabs, classlabs = averageframes(transformed_data_dict[key], audiolabels, labels)
            transformed_data_dict[key] = average_data
        data_list.append(transformed_data_dict['none'])
        pcadata_list.append(transformed_data_dict['pca'])
        ldadata_list.append(transformed_data_dict['lda'])
        nmfdata_list.append(transformed_data_dict['nmf'])
        ssnmfdata_list.append(transformed_data_dict['ssnmf'])
    return data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs


def write_output(data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs):
    pickle.dump([ldadata_list, classlabs, audiolabs], open(OUTPUT_FILES[0], 'wb'))
    pickle.dump([pcadata_list, classlabs, audiolabs], open(OUTPUT_FILES[1], 'wb'))
    pickle.dump([nmfdata_list, classlabs, audiolabs], open(OUTPUT_FILES[2], 'wb'))
    pickle.dump([ssnmfdata_list, classlabs, audiolabs], open(OUTPUT_FILES[3], 'wb'))
    pickle.dump([data_list, classlabs, audiolabs], open(OUTPUT_FILES[4], 'wb'))
    

if __name__ == '__main__':
    # first only lda - because it goes fast
    data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs = lda_map_and_average_frames(min_variance=0.99)    
    write_output(data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs)
    # then add nmf,ssnmf    
    #data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs = map_and_average_frames(min_variance=0.99)
    #write_output(data_list, pcadata_list, ldadata_list, nmfdata_list, ssnmfdata_list, classlabs, audiolabs)
    