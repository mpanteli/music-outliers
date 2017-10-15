# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:52 2017

@author: mariapanteli
"""

import pytest

import numpy as np
import os

import scripts.OPMellin as OPMellin


opm = OPMellin.OPMellin()
TEST_AUDIO_FILE = os.path.join(os.path.dirname(__file__), 'data', 'mel_1_2_1.wav')

def test_load_audiofile():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    assert opm.y is not None and opm.sr is not None


def test_mel_spectrogram():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    opm.mel_spectrogram(y=opm.y, sr=opm.sr)
    # assume 40 mel bands
    assert opm.melspec.shape[0] == 40


def test_post_process_spec():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    opm.mel_spectrogram(y=opm.y, sr=opm.sr)
    melspec = opm.melspec
    opm.post_process_spec(melspec=melspec)
    proc_melspec = opm.melspec
    assert melspec.shape == proc_melspec.shape


def test_onset_patterns_n_frames():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    opm.mel_spectrogram(y=opm.y, sr=opm.sr)
    opm.onset_patterns(melspec=opm.melspec, melsr=opm.melsr)
    assert opm.op.shape[2] == np.round(((opm.melspec.shape[1] / opm.melsr) - opm.win2sec) * 2.)


def test_onset_patterns_n_bins():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    opm.mel_spectrogram(y=opm.y, sr=opm.sr)
    opm.onset_patterns(melspec=opm.melspec, melsr=opm.melsr)
    assert opm.op.shape[0] == 40    


def test_post_process_op():
    audiofile = TEST_AUDIO_FILE
    opm.load_audiofile(audiofile, segment=False)
    opm.mel_spectrogram(y=opm.y, sr=opm.sr)
    opm.onset_patterns(melspec=opm.melspec, melsr=opm.melsr)
    op = opm.op
    opm.post_process_op()
    proc_op = opm.op
    assert op.shape == proc_op.shape