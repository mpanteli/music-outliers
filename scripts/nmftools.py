# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:46:13 2016

@author: https://github.com/keik/nmftools/blob/master/nmftools/core.py
"""

import numpy as np


def nmf(Y, R=3, n_iter=50, init_H=[], init_U=[], verbose=False):
    """
    decompose non-negative matrix to components and activation with NMF
    Y ≈ HU
    Y ∈ R (m, n)
    H ∈ R (m, k)
    HU ∈ R (k, n)
    parameters
    ----
    Y: target matrix to decompose
    R: number of bases to decompose
    n_iter: number for executing objective function to optimize
    init_H: initial value of H matrix. default value is random matrix
    init_U: initial value of U matrix. default value is random matrix
    return
    ----
    Array of:
    0: matrix of H
    1: matrix of U
    2: array of cost transition
    """

    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0]
    N = Y.shape[1]

    # initialization
    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R,N);

    if len(init_H):
        H = init_H;
        R = init_H.shape[1]
    else:
        H = np.random.rand(M,R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)

    # computation of Lambda (estimate of Y)
    Lambda = np.dot(H, U)

    # iterative computation
    for i in range(n_iter):

        # compute euclid divergence
        cost[i] = euclid_divergence(Y, Lambda)

        # update H
        H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)

        # update U
        U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)

        # recomputation of Lambda
        Lambda = np.dot(H, U)

    return [H, U, cost]


def ssnmf(Y, R=3, n_iter=50, F=[], init_G=[], init_H=[], init_U=[], verbose=False):
    """
    decompose non-negative matrix to components and activation with semi-supervised NMF
    Y ≈ FG + HU
    Y ∈ R (m, n)
    F ∈ R (m, x)
    G ∈ R (x, n)
    H ∈ R (m, k)
    U ∈ R (k, n)
    parameters
    ----
    Y: target matrix to decompose
    R: number of bases to decompose
    n_iter: number for executing objective function to optimize
    F: matrix as supervised base components
    init_W: initial value of W matrix. default value is random matrix
    init_H: initial value of W matrix. default value is random matrix
    return
    ----
    Array of:
    0: matrix of F
    1: matrix of G
    2: matrix of H
    3: matrix of U
    4: array of cost transition
    """

    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0];
    N = Y.shape[1];
    X = F.shape[1]

    # initialization
    if len(init_G):
        G = init_G
        X = init_G.shape[1]
    else:
        G = np.random.rand(X, N)

    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R, N)

    if len(init_H):
        H = init_H
        R = init_H.shape[1]
    else:
        H = np.random.rand(M, R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)

    # computation of Lambda (estimate of Y)
    Lambda = np.dot(F, G) + np.dot(H, U)

    # iterative computation
    for it in range(n_iter):

        # compute euclid divergence
        cost[it] = euclid_divergence(Y, Lambda + eps)

        # update of H
        H *= (np.dot(Y, U.T) + eps) / (np.dot(np.dot(H, U) + np.dot(F, G), U.T) + eps)

        # update of U
        U *= (np.dot(H.T, Y) + eps) / (np.dot(H.T, np.dot(H, U) + np.dot(F, G)) + eps)

        # update of G
        G *= (np.dot(F.T, Y) + eps)[np.arange(G.shape[0])] / (np.dot(F.T, np.dot(H, U) + np.dot(F, G)) + eps)

        # recomputation of Lambda (estimate of V)
        Lambda = np.dot(H, U) + np.dot(F, G)

    return [F, G, H, U, cost]


def euclid_divergence(V, Vh):
    d = 1 / 2 * (V ** 2 + Vh ** 2 - 2 * V * Vh).sum()
    return d