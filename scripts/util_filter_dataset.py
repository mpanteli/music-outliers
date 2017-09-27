# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:44:48 2017

@author: mariapanteli
"""

import os
import numpy as np
import pandas as pd


def get_speech_vamp(df):
    jspeech = df.columns.get_loc("Speech")
    nfiles = len(df)
    speechinds = []
    for i in range(nfiles):
        #print i
        if os.path.exists(df.iat[i, jspeech]) and os.path.getsize(df.iat[i, jspeech])>0:
            bounds = pd.read_csv(df.iat[i, jspeech], header=None, delimiter='\t').get_values()
            if len(bounds)>0:
                if len(np.where(bounds[:,2]=='m')[0])==0 or len(np.where(bounds[:,2]=='s')[0])==len(bounds):
                    speechinds.append(i)
    return speechinds


def get_speech_meta(df):
    genres = np.array(df["Genre_Album"].get_values(), dtype=str)
    speechinds_genre = []
    invalid_genres = ["Spoken Word", "Language Instruction", "Classical", 
                        "Poetry", "Nature|Sounds", "Music Instruction", 
                        "Soundtracks &amp", "Contemporary &amp", "Jazz &amp",
                        "Sounds", "Ragtime", "Nature", "Electronic",
                        "African American Spoken", "Blues", "Gospel",
                        "Psychology &amp"]
    for i in range(len(genres)):
        genre = genres[i]
        #if genre in invalid_genres:
        if any(x in genre for x in invalid_genres):
            speechinds_genre.append(i)
    return speechinds_genre


def get_missing_csv(df):
    nfiles = len(df)
    missing_csv = []
    for i in range(nfiles):
        if not (os.path.exists(df["Melspec"].iloc[i]) and os.path.exists(df["Chroma"].iloc[i]) and os.path.exists(df["Melodia"].iloc[i])):
            missing_csv.append(i)
    return missing_csv


def get_missing_country_meta(df):
    nfiles = len(df)
    missing_country = []
    country_labels = np.array(df['Country'].get_values(), dtype=str)
    invalid_countries = ['Unidentified', 'unknown', 'nan', 
                         'Yugoslavia (former)', 'Pathian village  Wangulei ', 
                         'Joulouloum  either Senegal or The Gambia ']
    for i in range(nfiles):
        country = country_labels[i]
        if country in invalid_countries:
            missing_country.append(i)
    return missing_country


def remove_missing_data(df):
    speechinds_vamp = get_speech_vamp(df)
    speechinds_genre = get_speech_meta(df)
    speechinds = set(speechinds_vamp) | set(speechinds_genre)
    missing = set(get_missing_csv(df))
    missing_country = set(get_missing_country_meta(df))
    selectinds = np.asarray(list(set(range(len(df))) - (missing | speechinds | missing_country)))

    df = df.iloc[selectinds, :]
    return df