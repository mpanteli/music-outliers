# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np

import scripts.map_and_average as map_and_average


def test_remove_inds():
    labels = np.array(['a', 'a', 'b', 'unknown'])
    features = np.array([[0, 1], [0,2], [0, 3], [0, 4]])
    audiolabels = np.array(['a', 'b', 'c', 'd'])
    features, labels, audiolabels = map_and_average.remove_inds(features, labels, audiolabels)
    assert len(features) == 3 and len(labels) == 3 and len(audiolabels) == 3


def test_remove_inds():
    labels = np.array(['a', 'a', 'b', 'unknown'])
    features = np.array([[0, 1], [0,2], [0, 3], [0, 4]])
    audiolabels = np.array(['a', 'b', 'c', 'd'])
    features, labels, audiolabels = map_and_average.remove_inds(features, labels, audiolabels)
    features_true = np.array([[0, 1], [0,2], [0, 3]])
    assert np.array_equal(features, features_true)


def test_averageframes():
    classlabels = np.array(['a', 'a', 'b', 'b', 'b'])
    features = np.array([[0, 1], [0,2], [0, 1], [1, 1], [2, 1]])
    audiolabels = np.array(['a', 'a', 'b', 'b', 'b'])
    feat, audio, labels = map_and_average.averageframes(features, audiolabels, classlabels)
    feat_true = np.array([[0, 1.5], [1, 1]])
    assert np.array_equal(feat, feat_true)


def test_limit_to_n_seconds():
    X = np.random.randn(10, 3)
    Y = np.random.randn(10)
    Yaudio = np.concatenate([np.repeat('a', 7), np.repeat('b', 3)])
    Xn, Yn, Yaudion = map_and_average.limit_to_n_seconds([X, Y, Yaudio], n_sec=3.0, win_sec=0.5)
    Yaudion_true = np.concatenate([np.repeat('a', 5), np.repeat('b', 3)])
    assert np.array_equal(Yaudion_true, Yaudion) and len(Xn)==len(Yn) and len(Yn)==len(Yaudion)