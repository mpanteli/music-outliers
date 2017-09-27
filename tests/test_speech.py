# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:57:53 2017

@author: mariapanteli
"""

import numpy as np
import pickle
import os

# Should really use package resources
DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'lda_data_melodia_8.pickle')


X_list, Y, Yaudio = pickle.load(open(DATA_FILE,'rb'))

uniq_audio, uniq_counts = np.unique(Yaudio, return_counts=True)
# this should be applied to /import/c4dm-04/mariap/train_data_melodia_8.pickle

# instead test the get music_idx bounds, seems to ignore segments less than 8 seconds, but confirm..