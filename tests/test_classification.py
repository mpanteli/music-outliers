# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np
from sklearn.model_selection import train_test_split

import scripts.classification as classification


def test_confusion_matrix():
    X = np.random.randn(100, 3)
    # create 2 classes by shifting the entries of half the samples
    X[-50:, :] = X[-50:, :] + 10
    Y = np.concatenate([np.repeat('a', 50), np.repeat('b', 50)])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6, random_state=1, stratify=Y)
    accuracy, _ = classification.confusion_matrix(X_train, Y_train, X_test, Y_test)
    # expect perfect accuracy for this 'easy' dataset
    assert accuracy == 1.0

