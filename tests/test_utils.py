# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np
import pandas as pd
import pickle
import os

import scripts.utils as utils


def test_get_outliers():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    # create outliers by shifting the entries of the last 5 samples
    X[-5:, :] = X[-5:, :] + 10
    Y = np.concatenate([np.repeat('a', 95), np.repeat('b', 5)])
    threshold, y_pred, MD = utils.get_outliers(X)
    # expect that items from country 'b' are detected as outliers
    assert np.array_equal(y_pred[-5:], np.ones(5))


def test_get_outliers():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    # create outliers by shifting the entries of the last 5 samples
    X[-5:, :] = X[-5:, :] + 10
    Y = np.concatenate([np.repeat('a', 95), np.repeat('b', 5)])
    threshold, y_pred, MD = utils.get_outliers_Mahal(X)
    # expect that items from country 'b' are detected as outliers
    assert np.array_equal(y_pred[-5:], np.ones(5))


def test_pca_data():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    X[-5:, :] = X[-5:, :] + 10
    X_pca, n_pc = utils.pca_data(X, min_variance=0.8)
    assert n_pc < X.shape[1]


def test_get_local_outliers_from_neighbors_dict():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    n_outliers = 3
    X[-n_outliers:, :] = X[-n_outliers:, :] + 10
    Y = np.concatenate([np.repeat('a', 20), np.repeat('b', 20), np.repeat('c', 20), 
                        np.repeat('k', 20), np.repeat('l', 20)])
    w_dict = {'a': ['b', 'c'], 'b': ['a', 'c'], 'c': ['b', 'a'], 'k': ['l'], 'l':['k']}
    spatial_outliers = utils.get_local_outliers_from_neighbors_dict(X, Y, w_dict)
    # last n samples of 'l' country must be outliers
    assert np.array_equal(spatial_outliers[-1][3][-n_outliers:], np.ones(n_outliers))


def test_best_n_clusters_silhouette():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    X[:30, :] = X[:30, :] + 10
    X[-30:, :] = X[-30:, :] + 20
    bestncl, _ = utils.best_n_clusters_silhouette(X, max_ncl=10)
    assert bestncl == 3

