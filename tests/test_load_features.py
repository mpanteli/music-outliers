# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import os
import numpy as np
import pandas as pd

import scripts.load_features as load_features

feat_loader = load_features.FeatureLoader(win2sec=8)

TEST_METADATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'metadata.csv')
TEST_MELODIA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'melodia_mel_1_2_1.csv')

def test_get_music_idx_from_bounds():
    bounds = np.array([['0', '10.5', 'm']])
    sr = feat_loader.framessr2            
    music_bounds = feat_loader.get_music_idx_from_bounds(bounds, sr=sr)
    # upper bound minus half window size
    #half_win_sec = 4.0  # assume 8-second window
    win_sec = 8
    music_bounds_true = np.arange(np.round(sr * (np.float(bounds[-1, 1]) - win_sec)), dtype=int)
    assert np.array_equal(music_bounds, music_bounds_true)
    
    
def test_get_music_idx_from_bounds_short_segment():
    # anything less than half window size is not processed
    bounds = np.array([['0', '7.9', 'm']])
    sr = feat_loader.framessr2
    music_bounds = feat_loader.get_music_idx_from_bounds(bounds, sr=sr)
    music_bounds_true = np.array([])
    assert np.array_equal(music_bounds, music_bounds_true)


def test_get_music_idx_from_bounds_single_frame():
    bounds = np.array([['0', '8.1', 'm']])
    sr = feat_loader.framessr2
    music_bounds = feat_loader.get_music_idx_from_bounds(bounds, sr=sr)
    music_bounds_true = np.array([0])
    assert np.array_equal(music_bounds, music_bounds_true)


def test_get_music_idx_from_bounds_mix_segments():
    bounds = np.array([['0', '10.5', 'm'], 
              ['10.5', '3.0', 's'],
              ['13.5', '5.0', 'm']])
    sr = feat_loader.framessr2
    music_bounds = feat_loader.get_music_idx_from_bounds(bounds, sr=sr)
    #half_win_sec = 4.0  # assume 8-second window
    win_sec = 8.0  # assume 8-second window
    music_bounds_true = np.concatenate([np.arange(np.round(sr * (10.5 - win_sec)), dtype=int),
                                        np.arange(np.round(sr * 13.5), 
                                            np.round(sr * (18.5 - win_sec)), dtype=int)])
    assert np.array_equal(music_bounds, music_bounds_true)
    

def test_get_music_idx_from_bounds_overlap_segments():
    bounds = np.array([['0', '10.5', 'm'], 
              ['9.5', '3.0', 's'],
              ['11.5', '5.0', 'm']])
    sr = feat_loader.framessr2
    music_bounds = feat_loader.get_music_idx_from_bounds(bounds, sr=sr)
    half_win_sec = 4.0  # assume 8-second window
    win_sec = 8.0  # assume 8-second window
    music_bounds_true = np.concatenate([np.arange(np.round(sr * (10.5 - win_sec)), dtype=int),
                                        np.arange(np.round(sr * 11.5), 
                                            np.round(sr * (16.5 - win_sec)), dtype=int)])
    assert np.array_equal(music_bounds, music_bounds_true)


def test_average_local_frames():
    frames = np.array([[0, 0.5, 1], [1, 1, 1]])
    aveframes = feat_loader.average_local_frames(frames)
    aveframes_true = np.array([[0.5], [1]])
    assert np.array_equal(aveframes, aveframes_true)


def test_average_local_frames_multiple_frames():
    frames1 = np.concatenate([np.repeat(0.5, feat_loader.hop2), np.repeat(0, feat_loader.win2)])
    frames2 = np.concatenate([np.repeat(1.5, feat_loader.hop2), np.repeat(1, feat_loader.win2)])
    frames = np.vstack([frames1, frames2])
    aveframes = feat_loader.average_local_frames(frames)
    aveframes_true = np.array([[0.5, 0], [1.5, 1]])
    # test only the second frame which contains values 0 or values 1 for all 8-second frame entries
    assert np.array_equal(aveframes[:, 1], aveframes_true[:, 1])


def test_average_local_frames_std():
    frames1 = np.concatenate([np.repeat(0.5, feat_loader.hop2), np.repeat(0, feat_loader.win2)])
    frames2 = np.concatenate([np.repeat(1.5, feat_loader.hop2), np.repeat(1, feat_loader.win2)])
    frames = np.vstack([frames1, frames2])
    aveframes = feat_loader.average_local_frames(frames, getstd=True)
    aveframes_true = np.array([[0.5, 0], [1.5, 1], [0.1, 0], [0.1, 0]])
    # test only the second frame which contains values 0 or values 1 for all 8-second frame entries
    assert np.array_equal(aveframes[:, 1], aveframes_true[:, 1])


def test_get_op_from_melspec_n_frames():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    opmel = feat_loader.get_op_from_melspec(melspec)
    n_frames = opmel.shape[0]
    # expect 4 frames for windows not centered and .5 sec hop size
    # np.round((dur_sec - feat_loader.win2sec) * feat_loader.framessr2)
    assert n_frames == 4


def test_get_op_from_melspec_n_bins():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    opmel = feat_loader.get_op_from_melspec(melspec)
    n_bins = opmel.shape[1]
    assert n_bins == 40 * 200


def test_get_op_from_melspec_K_bands():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    K = 2
    opmel = feat_loader.get_op_from_melspec(melspec, K=K)
    n_bins = opmel.shape[1]
    assert n_bins == K * 200


def test_get_mfcc_from_melspec_n_coef():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    mfcc = feat_loader.get_mfcc_from_melspec(melspec, deltamfcc=False, avelocalframes=False)
    assert mfcc.shape[1] == 20


def test_get_mfcc_from_melspec_n_coef_delta():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    mfcc = feat_loader.get_mfcc_from_melspec(melspec, deltamfcc=True, avelocalframes=False)
    assert mfcc.shape[1] == 40


def test_get_mfcc_from_melspec_n_frames():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    mfcc = feat_loader.get_mfcc_from_melspec(melspec, deltamfcc=False, avelocalframes=False)
    assert mfcc.shape[0] == dur_frames


def test_get_mfcc_from_melspec_n_frames_win2():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    melspec = np.random.randn(40, dur_frames) # melspec with 40 melbands
    melspec = melspec - np.min(melspec)  # melspec must be positive
    mfcc = feat_loader.get_mfcc_from_melspec(melspec, deltamfcc=False, avelocalframes=True)
    n_frames_true = np.round((dur_sec - feat_loader.win2sec) * feat_loader.framessr2)
    assert mfcc.shape[0] == n_frames_true


def test_get_ave_chroma_align():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    chroma = np.random.randn(60, dur_frames) # chroma with 60 bins
    chroma = chroma - np.min(chroma)  # chroma must be positive
    ave_chroma = feat_loader.get_ave_chroma(chroma, alignchroma=True, avelocalframes=False)
    # the maximum bin across time is the first bin (after alignment)
    assert np.argmax(np.sum(ave_chroma, axis=0)) == 0
    

def test_get_ave_chroma_n_frames():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    chroma = np.random.randn(60, dur_frames) # chroma with 60 bins
    chroma = chroma - np.min(chroma)  # chroma must be positive
    ave_chroma = feat_loader.get_ave_chroma(chroma, avelocalframes=True, stdlocalframes=False)
    n_frames_true = np.round((dur_sec - feat_loader.win2sec) * feat_loader.framessr2)
    assert ave_chroma.shape[0] == n_frames_true


def test_get_ave_chroma_n_bins():
    dur_sec = 10.0
    dur_frames = np.int(np.round(dur_sec * feat_loader.framessr))
    np.random.seed(1)
    chroma = np.random.randn(60, dur_frames) # chroma with 60 bins
    chroma = chroma - np.min(chroma)  # chroma must be positive
    ave_chroma = feat_loader.get_ave_chroma(chroma, avelocalframes=True, stdlocalframes=True)
    assert ave_chroma.shape[1] == 120


def test_get_pb_for_file_empty():
    pbihist = feat_loader.get_pb_for_file('')
    assert np.array_equal(pbihist, [])
    

def test_get_pb_for_file_n_bins():
    pbihist = feat_loader.get_pb_for_file(TEST_MELODIA_FILE, nmfpb=False, scale=False)
    assert pbihist.shape[1] == 3600


def test_get_pb_for_file_align():
    pbihist = feat_loader.get_pb_for_file(TEST_MELODIA_FILE, nmfpb=False, scale=False)
    pbihist = pbihist.get_values()
    assert np.sum(pbihist[:, :60].ravel()) > np.sum(pbihist[:, 60:120].ravel())


def test_get_pb_for_file_nmf():
    pbihist = feat_loader.get_pb_for_file(TEST_MELODIA_FILE, nmfpb=True, scale=False)
    assert pbihist.shape[1] == 240


def test_get_features():
    df = pd.read_csv(TEST_METADATA_FILE)
    df = df.iloc[:1, :]
    data_list = feat_loader.get_features(df, precomp_melody=False)
    assert len(np.unique(data_list[-1])) == 1


def test_get_features_n_files():
    df = pd.read_csv(TEST_METADATA_FILE)
    n_files = 1
    df = df.iloc[:n_files, :]
    data_list = feat_loader.get_features(df, precomp_melody=False)
    assert len(np.unique(data_list[-1])) == n_files


def test_get_features_n_frames():
    df = pd.read_csv(TEST_METADATA_FILE)
    df = df.iloc[:1, :]
    data_list = feat_loader.get_features(df, precomp_melody=False)
    dur_sec = 11.5  # duration of first file in metadata.csv is > 11 seconds
    n_frames_true = np.round((dur_sec - feat_loader.win2sec) * feat_loader.framessr2)
    assert len(data_list[0]) == n_frames_true
    