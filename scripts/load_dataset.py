# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:52:57 2017

@author: mariapanteli
"""
import os

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import extract_primary_features
import load_features
import util_filter_dataset


WIN_SIZE = 8
DATA_DIR = 'data'
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.csv')
OUTPUT_FILES = [os.path.join(DATA_DIR, 'train_data_'+str(WIN_SIZE)+'.pickle'), 
                os.path.join(DATA_DIR, 'val_data_'+str(WIN_SIZE)+'.pickle'), 
                os.path.join(DATA_DIR, 'test_data_'+str(WIN_SIZE)+'.pickle')]


def get_train_val_test_idx(X, Y, seed=None):
    """ Split in train, validation, test sets.
    
    Parameters
    ----------
    X : np.array
        Data or indices.
    Y : np.array
        Class labels for data in X.
    seed: int
        Random seed.
    Returns
    -------
    (X_train, Y_train) : tuple
        Data X and labels y for the train set
    (X_val, Y_val) : tuple
        Data X and labels y for the validation set
    (X_test, Y_test) : tuple
        Data X and labels y for the test set
    
    """
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.6, random_state=seed, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.5, random_state=seed, stratify=Y_val_test)
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def subset_labels(Y, N_min=10, N_max=100, seed=None):
    """ Subset dataset to contain minimum N_min and maximum N_max instances 
        per class. Return indices for this subset. 
    
    Parameters
    ----------
    Y : np.array
        Class labels
    N_min : int
        Minimum instances per class
    N_max : int
        Maximum instances per class
    seed: int
        Random seed.
    
    Returns
    -------
    subset_idx : np.array
        Indices for a subset with classes of size bounded by N_min, N_max
    
    """
    np.random.seed(seed=seed)
    subset_idx = []
    labels = np.unique(Y)
    for label in labels:
        label_idx = np.where(Y==label)[0]
        counts = len(label_idx)
        if counts>=N_max:
            subset_idx.append(np.random.choice(label_idx, N_max, replace=False))
        elif counts>=N_min and counts<N_max:
            subset_idx.append(label_idx)
        else:
            # not enough samples for this class, skip
            print("Found only %s samples from class %s (minimum %s)" % (counts, label, N_min))
            continue
    if len(subset_idx)>0:
        subset_idx = np.concatenate(subset_idx, axis=0)
        return subset_idx
    else:
        raise ValueError('No classes found with minimum %s samples' % N_min)


def check_extract_primary_features(df):
    """ Check if csv files for melspectrograms, chromagrams, melodia, speech/music segmentation exist, 
        if the csv files don't exist, extract them.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata including class label and path to audio, melspec, chroma
    """
    extract_melspec, extract_chroma, extract_melodia, extract_speech = False, False, False, False
    # TODO: Check for each file in df, instead of just the first one
    if os.path.exists(df['Audio'].iloc[0]):
        if not os.path.exists(df['Melspec'].iloc[0]):
            extract_melspec = True
        if not os.path.exists(df['Chroma'].iloc[0]):
            extract_chroma = True
        if not os.path.exists(df['Melodia'].iloc[0]):
            extract_melodia = True
        if not os.path.exists(df['Speech'].iloc[0]):
            extract_speech = True
    else:
        print("Audio file %s does not exist. Primary features will not be extracted." % df['Audio'].iloc[0])
    extract_primary_features.extract_features_for_file_list(df, 
        melspec=extract_melspec, chroma=extract_chroma, melodia=extract_melodia, speech=extract_speech)


def extract_features(df, win2sec=8.0):
    """ Extract features from melspec and chroma.
    
    Parameters
    ----------
    df : pd.DataFrame
        Metadata including class label and path to audio, melspec, chroma
    win2sec : float
        The window size for the second frame decomposition of the features
        
    Returns
    -------
    X : np.array
        The features for every frame x every audio file in the dataset
    Y : np.array
        The class labels for every frame in the dataset
    Y_audio : np.array
        The audio labels
    """
    feat_loader = load_features.FeatureLoader(win2sec=win2sec)
    frames_rhy, frames_mfcc, frames_chroma, frames_mel, Y_df, Y_audio_df = feat_loader.get_features(df, precomp_melody=False)
    print frames_rhy.shape, frames_mel.shape, frames_mfcc.shape, frames_chroma.shape
    X = np.concatenate((frames_rhy, frames_mel, frames_mfcc, frames_chroma), axis=1)
    Y = Y_df.get_values()
    Y_audio = Y_audio_df.get_values()
    return X, Y, Y_audio


def sample_dataset(df):
    """ Select min 10 - max 100 recs from each country.

    Parameters
    ----------
    df : pd.DataFrame
        The metadata (including country) of the tracks.

    Returns
    -------
    df : pd.DataFrame
        The metadata for the selected subset of tracks.
    """
    df = util_filter_dataset.remove_missing_data(df)
    subset_idx = subset_labels(df['Country'].get_values())
    df = df.iloc[subset_idx, :]
    return df
    

def features_for_train_test_sets(df, write_output=False):
    """Split in train/val/test sets, extract features and write output files.

    Parameters
    -------
    df : pd.DataFrame
        The metadata for the selected subset of tracks.
    write_output : boolean
        Whether to write files with the extracted features for train/val/test sets.
    """
    X_idx, Y = np.arange(len(df)), df['Country'].get_values()
    extract_features(df.iloc[np.array([0])], win2sec=WIN_SIZE)
    train_set, val_set, test_set = get_train_val_test_idx(X_idx, Y)
    X_train, Y_train, Y_audio_train = extract_features(df.iloc[train_set[0]], win2sec=WIN_SIZE)
    X_val, Y_val, Y_audio_val = extract_features(df.iloc[val_set[0]], win2sec=WIN_SIZE)   
    X_test, Y_test, Y_audio_test = extract_features(df.iloc[test_set[0]], win2sec=WIN_SIZE)
   
    train = [X_train, Y_train, Y_audio_train]
    val = [X_val, Y_val, Y_audio_val]
    test = [X_test, Y_test, Y_audio_test]
    if write_output:
        with open(OUTPUT_FILES[0], 'wb') as f:
            pickle.dump(train, f)            
        with open(OUTPUT_FILES[1], 'wb') as f:
            pickle.dump(val, f)
        with open(OUTPUT_FILES[2], 'wb') as f:
            pickle.dump(test, f)
    return train, val, test


if __name__ == '__main__':
    # load dataset
    df = pd.read_csv(METADATA_FILE)
    check_extract_primary_features(df)
    df = sample_dataset(df)
    train, val, test = features_for_train_test_sets(df, write_output=True)

