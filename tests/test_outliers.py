# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np

import scripts.outliers as outliers


def test_country_outlier_df():
    counts = {'a':2, 'b':3}
    labels = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b'])
    df = outliers.country_outlier_df(counts, labels, normalize=True)
    assert np.array_equal(df['Outliers'].get_values(), np.array([0.5, 1.0]))


def test_normalize_outlier_counts():
    outlier_counts = {'a':2, 'b':3}
    country_counts = {'a':4, 'b':3}
    outlier_counts = outliers.normalize_outlier_counts(outlier_counts, country_counts)
    outlier_counts_true = {'a':.5, 'b':1.}
    assert np.array_equal(outlier_counts, outlier_counts_true)


def test_get_outliers_df():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    # create outliers by shifting the entries of the last 5 samples
    X[-5:, :] = X[-5:, :] + 10
    Y = np.concatenate([np.repeat('a', 95), np.repeat('b', 5)])
    df, threshold, MD = outliers.get_outliers_df(X, Y)
    # expect that items from country 'b' are detected as outliers
    assert np.array_equal(df['Outliers'].get_values(), np.array([0., 1.0]))

