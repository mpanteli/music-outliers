# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:06:12 2016

@author: mariapanteli
"""

import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import HoverTool, TapTool, CustomJS
from bokeh.plotting import figure, show, save, output_file, ColumnDataSource
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon
import random
from bokeh.models.widgets import Panel, Tabs
import os


SHAPEFILE = os.path.join(os.path.dirname(__file__), 'util_data', 'shapefiles', 'ne_110m_admin_0_countries')


def get_random_point_in_polygon(poly):
    '''Select at random a point within given polygon boundaries.

    Parameters
    ----------
    poly : Polygon
        The polygon boundaries.

    Returns
    -------
    p : Point
        A random point (x, y coords) inside the given polygon.
    '''
    (minx, miny, maxx, maxy) = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


def get_random_point_in_country_poly(countries_data):
    '''Load country polygons and selects a point at random within each polygon.

    Parameters
    ----------
    countries_data : np.array, 1D
        Names of countries to select random points. 

    Returns
    -------
    data_x : list of float
        The x-coordinates of random points within each country in countries_data.
    data_y : list of float
        The y-coordinates of random points within each country in countries_data.
    '''
    pp_x, pp_y, coords_poly, countries_poly = get_countries_lonlat_poly(SHAPEFILE)
    data_x = []
    data_y = []
    for country in countries_data:
        #print country
        poly_inds = np.where(countries_poly==country)[0]
        if len(poly_inds)<1:
            data_x.append(np.nan)
            data_y.append(np.nan)
            continue
        poly = coords_poly[poly_inds[0]]
        if len(poly_inds)>1:
            # if many polys for country choose the largest one (ie most points)
            len_list = [len(pp_x[poly_ind]) for poly_ind in poly_inds]
            poly = coords_poly[poly_inds[np.argmax(len_list)]]
        p = Polygon(poly)
        point_in_poly = get_random_point_in_polygon(p)
        data_x.append(point_in_poly.x)
        data_y.append(point_in_poly.y)
    return data_x, data_y


def get_countries_lonlat_poly(shapefile):
    '''Load spatial information for each country from shapefiles.

    Parameters
    ----------
    shapefile : str
        Path to shapefile.

    Returns
    -------
    pp_x : list of float
        The x-coordinates of country polygons.
    pp_y : list of float
        The y-coordinates of country polygons.
    mm.units : list of float tuples.
        Polygon coordinates for each country.
    countries_poly : np.arry, 1D
        Country names for each polygon.
    '''
    mm=Basemap()
    mm.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
    pp_x = []
    pp_y = []
    for shape in mm.units:
        pp_x.append([ss[0] for ss in shape])
        pp_y.append([ss[1] for ss in shape])
    countries_poly = []
    for mm_info in mm.units_info:
        countries_poly.append(mm_info['admin'])
    countries_poly = np.array(countries_poly, dtype=str)
    #(-52.55642473001839, 2.504705308437053) for French Guiana
    countries_poly[102] = 'French Guiana'  # manual correction
    return pp_x, pp_y, mm.units, countries_poly


def add_bokeh_interactivity(p, r, hover_outlier=False):
    '''Add plot interactivity.
    '''
    callback = CustomJS(args=dict(r=r), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        var d1 = cb_obj.get('data');
        url = d1['url'][inds[0]];
        if (url){
            window.open(url);}""")
    hover_tooltips = """
        <div>
            <div><span style="font-size: 17px; font-weight: bold;">@name</span></div>
            <div><span style="font-size: 12px;">@info</span></div>
        </div>"""    
    hover_tooltips_outlier = """
        <div>
            <div><span style="font-size: 17px; font-weight: bold;">@name</span></div>
            <div><span style="font-size: 12px;">@info</span></div>
            <div><span style="font-size: 10px; color: #500;">@outlierMD</span></div>
            <div><span style="font-size: 12px;">@collection</span></div>
        </div>"""
    if hover_outlier:
        p.add_tools(HoverTool(renderers=[r], tooltips=hover_tooltips_outlier))
    else:
        p.add_tools(HoverTool(renderers=[r], tooltips=hover_tooltips))
    p.add_tools(TapTool(renderers=[r], callback = callback))
    return p


def beautify_bokeh_background(p):
    '''Remove unnecessary background in plot. 
    '''
    p.outline_line_color = None
    p.grid.grid_line_color=None
    p.axis.axis_line_color=None
    p.axis.major_label_text_font_size='0pt'
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    return p

    
def plot_outliers_world_figure(MD, y_pred, df, out_file=None):
    '''Visualise outliers on an interactive world map figure. 

    Parameters
    ----------
    MD : np.array, float, 1D
        Mahalanobis distances for each data point.
    y_pred : np.array, boolean, 1D
        Whether data point was detected as an outlier or not. 
    df : pd.DataFrame
        Additional metadata (country, culture, language, genre, collection) for each data point. 
    out_file : str
        Path to export html file.

    Returns
    -------
    p : bokeh
        The interactive map.
    '''
    pp_x, pp_y, coords_poly, countries_poly = get_countries_lonlat_poly(SHAPEFILE)    
    data_x, data_y = get_random_point_in_country_poly(df['Country'].get_values())    

    alpha_color = (MD-np.min(MD)+0.5)/(np.max(MD)-np.min(MD)+0.5)
    alpha_color[y_pred==False] = 0.3
    
    circle_color = np.repeat('grey', repeats=len(y_pred))
    circle_color[y_pred] = 'red'
    
    outlier_info = []
    for i in range(len(MD)):
        if y_pred[i]:
            # if outlier
            outlier_info.append('outlier, MD=' + str(int(MD[i])))
        else:
            outlier_info.append('non-outlier, MD=' + str(int(MD[i])))

    source = ColumnDataSource(data=dict(
        x=data_x,
        y=data_y,
        name=df['Country'].get_values(),
        color=circle_color,
        alpha=alpha_color,
        info = zip(df['Culture'].get_values(),df['Language'].get_values(),df['Genre'].get_values()),
        outlierMD = outlier_info,
        collection = df['Collection'].get_values(),
        url=df['Url'].get_values()
    ))
    
    TOOLS="wheel_zoom,box_zoom,pan,reset,save"
    
    p = figure(tools=TOOLS, plot_width=1200, title="Outlier recordings per country (click on each point to listen to the audio). More info at: mpanteli.github.io/music-outliers/demo/", title_text_font_size='14pt')
    outlier_ind = np.argmax(MD)
    nonoutlier_ind = np.argmin(MD)
    rleg1 = p.circle(data_x[outlier_ind], data_y[outlier_ind], fill_color='red', alpha=alpha_color[outlier_ind], size=6,
                     line_color=None, selection_color="firebrick", nonselection_color='white', legend="outliers")
    rleg2 = p.circle(data_x[nonoutlier_ind], data_y[nonoutlier_ind], fill_color='grey', alpha=alpha_color[nonoutlier_ind],
                     size=6, line_color=None, selection_color="firebrick", nonselection_color='white', legend="non-outliers")  
    r1 = p.patches(pp_x, pp_y, fill_color='white', line_width=0.4, line_color='grey')
    r2 = p.circle_cross('x','y', fill_color='color', alpha='alpha', size=6, line_color=None, 
                        selection_color="firebrick", nonselection_color='color', source=source) 
    
    p = add_bokeh_interactivity(p, r2, hover_outlier=True)
    p = beautify_bokeh_background(p)
    
    if out_file is not None:
        output_file(out_file)
        save(p)
    #show(p)
    return p


def plot_tabs(tab_all, tabs_feat, out_file=None):
    '''Add tabs to bokeh plot.
    '''
    tab1 = Panel(child=tab_all, title="All")
    tab2 = Panel(child=tabs_feat[0], title="Rhythm")
    tab3 = Panel(child=tabs_feat[1], title="Melody")
    tab4 = Panel(child=tabs_feat[2], title="Timbre")
    tab5 = Panel(child=tabs_feat[3], title="Harmony")
    tabs = Tabs(tabs=[tab1,tab2,tab3,tab4,tab5])
    if out_file is not None:
        output_file(out_file)
        save(tabs)
    show(tabs)
