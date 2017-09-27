# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np

import scripts.load_dataset as load_dataset


def test_get_train_val_test_idx():
    X = np.arange(10)
    Y = np.concatenate([np.ones(5), np.zeros(5)])
    train, val, test = load_dataset.get_train_val_test_idx(X, Y, seed=1)
    assert len(train[0]) == 6 and len(val[0]) == 2 and len(test[0]) == 2


def test_get_train_val_test_idx_stratify():
    X = np.arange(10)
    Y = np.concatenate([np.ones(5), np.zeros(5)])
    train, val, test = load_dataset.get_train_val_test_idx(X, Y, seed=1)
    assert np.array_equal(np.unique(train[1]), np.unique(val[1]))


def test_subset_labels():
    Y = np.concatenate([np.ones(5), 2*np.ones(10), 3*np.ones(100)])
    subset_idx = load_dataset.subset_labels(Y, seed=1)
    subset_idx = np.sort(subset_idx)
    subset_idx_true = np.arange(5, 115)
    assert np.array_equal(subset_idx, subset_idx_true)

    