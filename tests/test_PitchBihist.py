# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np
import os

import scripts.PitchBihist as PitchBihist


pbi = PitchBihist.PitchBihist()
TEST_MELODIA_FILE = os.path.join(os.path.dirname(__file__), os.path.pardir, 
                                 'data', 'sample_dataset', 'Melodia', 'mel_1_2_1.csv')

def test_hz_to_cents():
    freq_Hz = np.array([32.703, 65.406, 55, 110])
    freq_cents = pbi.hz_to_cents(freq_Hz)
    freq_cents_true = np.array([0, 1200, 900, 2100])
    assert np.array_equal(freq_cents, freq_cents_true)


def test_wrap_to_octave():
    cents = np.array([900, 2100, 1200])
    octave_cents = pbi.wrap_to_octave(cents)
    octave_cents_true = np.array([900, 900, 0])
    assert np.array_equal(octave_cents, octave_cents_true)


def test_get_melody_from_file():
    melodia_file = TEST_MELODIA_FILE
    melody = pbi.get_melody_from_file(melodia_file)
    assert len(melody) < 12. * pbi.melody_sr


def test_get_melody_matrix():
    melody = 440 * np.ones(1000)
    melody_matrix = pbi.get_melody_matrix(melody)
    n_frames = melody_matrix.shape[1]
    assert np.array_equal(melody_matrix[45, :], np.ones(n_frames))


def test_second_frame_decomposition():
    norigframes = 100
    nframes, win2, hop2 = pbi.second_frame_decomposition(norigframes)
    assert nframes == 1


def test_second_frame_decomposition():
    norigframes = np.ceil(pbi.melody_sr * 16.)
    nframes, win2, hop2 = pbi.second_frame_decomposition(norigframes)
    # for 16-sec segment with .5 hop size expect ~16 frames round up
    assert nframes == 17


def test_bihist_from_melodia():
    melodia_file = TEST_MELODIA_FILE
    bihist = pbi.bihist_from_melodia(melodia_file, secondframedecomp=False)
    assert bihist.shape == (60, 60)


def test_bihist_from_melodia_n_frames():
    melodia_file = TEST_MELODIA_FILE
    bihist = pbi.bihist_from_melodia(melodia_file, secondframedecomp=True)
    dur_sec = 11.5  # duration of first file in metadata.csv is > 11 seconds
    n_frames_true = np.round((dur_sec - pbi.win2sec) * 2)  # for .5 sec hop size
    assert bihist.shape[1] == n_frames_true


def test_bihistogram():
    melody = 440 * np.ones(1000)
    melody_matrix = pbi.get_melody_matrix(melody)
    bihist = pbi.bihistogram(melody_matrix, align=False)
    assert np.array_equal(bihist, np.zeros((60, 60)))


def test_bihistogram_values():
    melody = np.concatenate([440 * np.ones(500), 32.703 * np.ones(500)])
    melody_matrix = pbi.get_melody_matrix(melody)
    # melody transitions from A to C (bin 45/60 to bin 0/60)
    bihist = pbi.bihistogram(melody_matrix, align=False)
    # expect only element [45, 0] to be non-zero
    assert bihist[45, 0] > 0 and (np.sum(bihist) - bihist[45, 0]) == 0


def test_bihistogram_align():
    melody = np.concatenate([660 * np.ones(250), 440 * np.ones(500)])
    melody_matrix = pbi.get_melody_matrix(melody)
    # bin of max magnitude = A = 45/60
    bihist = pbi.bihistogram(melody_matrix, align=True)
    # bihist[20, 45] == 0
    # we shift up 45 so rows 20-45 and left 45 so cols 45-45
    assert bihist[20-45, 0] == 1
