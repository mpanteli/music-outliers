# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:35:51 2017

@author: mariapanteli
"""
import numpy as np
import pandas as pd
import json
import pysal # before shapely in util_plots
import fiona
import os
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.path.dirname(__file__), 'util_data')
JSON_DB = os.path.join(DATA_DIR, 'countries.json')
SHAPEFILE = os.path.join(DATA_DIR, 'shapefiles', 'ne_10m_admin_0_countries.shp')


def neighbors_from_json_file(data_countries, json_DB=JSON_DB):
    neighbors = {}
    with open(json_DB) as json_file:
        countries_dict = json.load(json_file)
    country_names = []
    country_iso = []
    country_borders_iso = []
    for country_info in countries_dict:
        country_names.append(country_info['name']['common'])
        country_iso.append(country_info['cca3'])
        country_borders_iso.append(country_info['borders'])
    # temporary fixes of country names to match json data
    country_names[country_names.index('United States')] = 'United States of America'
    country_names[country_names.index('Tanzania')] = 'United Republic of Tanzania'
    country_names[country_names.index('DR Congo')] = 'Democratic Republic of the Congo'
    country_names[country_names.index('Czechia')] = 'Czech Republic'
    for i, country in enumerate(data_countries):
        neighbors[i] = {} 
        if country in country_names:
            if len(country_borders_iso[country_names.index(country)])>0:
                # if country has neighbors according to json file
                neighbors_iso = country_borders_iso[country_names.index(country)]
                neighbors_names = [country_names[country_iso.index(nn)] for nn in neighbors_iso]
                for neighbor in neighbors_names:
                    if neighbor in data_countries:
                        neighbor_idx = np.where(data_countries==neighbor)[0][0]
                        neighbors[i][neighbor_idx] = 1.0                
    w = pysal.weights.W(neighbors, id_order=range(len(data_countries)))
    return w


def get_countries_from_shapefile(shapefile):
    shp = fiona.open(shapefile, 'r')
    countries = []
    if shp[0]["properties"].has_key("ADMIN"):
        country_keyword = "ADMIN"
    elif shp[0]["properties"].has_key("NAME"):
        country_keyword = "NAME"
    else:
        country_keyword = "admin"
    for line in shp:
        countries.append(line["properties"][country_keyword])
    shp.close()
    return countries


def replace_empty_neighbours_with_KNN(data_countries, w):
    shapefile = SHAPEFILE
    no_neighbors_idx = w.islands
    knn = 10
    wknn = pysal.knnW_from_shapefile(shapefile, knn)
    knn_countries = get_countries_from_shapefile(shapefile)
    neighbors = w.neighbors
    for nn_idx in no_neighbors_idx:
        country = data_countries[nn_idx]
        print country
        if country not in knn_countries:
            continue
        knn_country_idx = knn_countries.index(country)
        knn_country_neighbors = [knn_countries[nn] for nn in wknn.neighbors[knn_country_idx]]
        for knn_nn in knn_country_neighbors:
            if len(neighbors[nn_idx])>2:
                continue
            data_country_idx = np.where(data_countries==knn_nn)[0]
            if len(data_country_idx)>0:
                neighbors[nn_idx][data_country_idx[0]] = 1.0
    w = pysal.weights.W(neighbors, id_order=range(len(data_countries)))
    return w


def get_neighbors_for_countries_in_dataset(Y):
    # neighbors
    data_countries = np.unique(Y)
    w = neighbors_from_json_file(data_countries)
    w = replace_empty_neighbours_with_KNN(data_countries, w)
    return w, data_countries


def from_weights_to_dict(w, data_countries):
    w_dict = {}
    for i in w.neighbors:
        w_dict[data_countries[i]] = [data_countries[nn] for nn in w.neighbors[i]]
    return w_dict


def get_regions_from_shapefile(shapefile):
    shp = fiona.open(shapefile, 'r')
    countries = []
    regions = []
    if shp[0]["properties"].has_key("ADMIN"):
        country_keyword = "ADMIN"
        region_keyword = "REGION_WB"
    elif shp[0]["properties"].has_key("NAME"):
        country_keyword = "NAME"
        region_keyword = "REGION_WB"
    else:
        country_keyword = "admin"
        region_keyword = "region_wb"
    for line in shp:
        #countries.append(line["properties"]["admin"])
        countries.append(line["properties"][country_keyword])
        regions.append(line["properties"][region_keyword])
    shp.close()
    return countries, regions


def append_regions(df):
    countries, regions = get_regions_from_shapefile(SHAPEFILE)
    if 'French Guiana' not in countries:
        countries.append('French Guiana')
        regions.append('Latin America & Caribbean')
    df_regions = pd.DataFrame({'Country': countries, 'Region': regions})
    df_append = pd.merge(df, df_regions, how='left', on='Country')
    return df_append

def get_LH_HL_idx(lm, p_vals):
    sig_idx = np.where(p_vals<0.05)[0]  
    LH_idx = sig_idx[np.where(lm.q[sig_idx]==2)[0]]
    HL_idx = sig_idx[np.where(lm.q[sig_idx]==4)[0]]
    return LH_idx, HL_idx


def print_Moran_outliers(y, w, data_countries):
    lm = pysal.Moran_Local(y, w)
    p_vals = lm.p_z_sim
    LH_idx, HL_idx = get_LH_HL_idx(lm, p_vals)
    print 'LH', zip(data_countries[LH_idx], p_vals[LH_idx])  # LH
    print 'HL', zip(data_countries[HL_idx], p_vals[HL_idx])  # HL


def plot_Moran_scatterplot(y, w, data_countries, out_file=None):
    lm = pysal.Moran_Local(y, w)
    p_vals = lm.p_z_sim
    LH_idx, HL_idx = get_LH_HL_idx(lm, p_vals)
    
    ylt  = pysal.lag_spatial(lm.w, lm.y)
    yt = lm.y
    yt = (yt - np.mean(yt))/np.std(yt)
    ylt = (ylt - np.mean(ylt))/np.std(ylt)
    colors = plt.cm.spectral(np.linspace(0,1,5))
    quad = np.zeros(yt.shape, dtype=int)
    quad[np.bitwise_and(ylt > 0, yt > 0)]=1 # HH
    quad[np.bitwise_and(ylt > 0, yt < 0)]=2 # LH
    quad[np.bitwise_and(ylt < 0, yt < 0)]=3 # LL
    quad[np.bitwise_and(ylt < 0, yt > 0)]=4 # HL
    marker_color = colors[quad]
    marker_size = 40*np.ones(yt.shape, dtype=int)
    marker_size[LH_idx] = 140
    marker_size[HL_idx] = 140
    
    plt.figure()
    plt.scatter(yt, ylt, c=marker_color, s=marker_size, alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Spatially Lagged Value')
    plt.axvline(c='black', ls='--')
    plt.axhline(c='black', ls='--')
    plt.ylim(min(ylt)-0.5, max(ylt)+0.5)
    plt.xlim(min(yt)-0.5, max(yt)+1.5)
    for i in np.concatenate((LH_idx, HL_idx)):
        plt.annotate(data_countries[i], (yt[i], ylt[i]), xytext=(yt[i]*1.1, ylt[i]*1.1),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    extreme_points = np.concatenate(([np.argmin(ylt)], [np.argmax(ylt)], np.where(yt>np.mean(yt)+2.8*np.std(yt))[0], np.where(ylt>np.mean(yt)+2.8*np.std(ylt))[0]))
    extreme_points = np.array(list(set(extreme_points) - set(np.concatenate((LH_idx, HL_idx)))))
    for i in extreme_points:
        plt.annotate(data_countries[i], (yt[i]+0.1, ylt[i]))
    if out_file is not None:
        plt.savefig(out_file)
