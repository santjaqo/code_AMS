# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import h5py
import pandas as pd
import numpy as np
import os
import datetime
import fiona
from shapely.geometry import Polygon, Point
from rtree import index #please note that rtree has a dependancy on libspatialindex - a C library that most be properly running on your OS
import matplotlib.pyplot as plt
from descartes import PolygonPatch
# <codecell>
def rainfall_loader(year, month, day, hours, minutes):
    '''Reads a KNMI's h5 file and returns it into a Pandas DataFrame.
    Must be run in the folder containing the directory tree dowonloaded from KNMI's ftp.
    Note that rainfall values are in centesimas of mm. KNMI did this to avoid using decimals in the H5 files. Intensities refer to rainfall every 5min.
    
    Arguments:
    
    :year: -- string, in the format yyyy
    
    :month: -- string, in the format mm, e.g. 02 or 10
    
    :day: -- string, in the format dd
    
    :hours: -- string, in the format hh, e.g. 08 or 13
    
    :minute: -- string, in the format mm, e.g. 03 or 58
    
    '''
    saved_path = os.getcwd()
    os.chdir(year + '/' + month + '/') #format must be yyyy, mm
    KNMI_25RAC_h5_filename = 'RAD_NL25_RAC_MFBS_5min_'+year+month+day+hours+minutes+'_NL.h5'
    loaded_file = h5py.File(KNMI_25RAC_h5_filename, 'r')
    os.chdir(saved_path)
    loaded_df = pd.DataFrame(loaded_file['image1']['image_data'][:])
    loaded_file.close()
    loaded_df.replace(65535, np.nan, inplace=True)
    loaded_df = loaded_df/100
    i_rows = np.linspace(-3650.5, -4414.5, num=765)
    i_columns = np.linspace(0.5, 699.5, num=700)
    loaded_df.columns = i_columns
    loaded_df.index = i_rows
    loaded_df_stacked = loaded_df.stack()
    loaded_df_stacked = pd.DataFrame(loaded_df_stacked.reset_index())
    loaded_df_stacked.columns = ['Y', 'X', 'value']
    loaded_df_stacked['timestamp'] = datetime.datetime(int(year), int(month), int(day), int(hours), int(minutes))
    loaded_df_stacked['sequence'] = np.arange(len(loaded_df_stacked))
    return loaded_df_stacked
# <codecell>
def timer(zero_time, final_time, time_step):
    '''Returns a list of tuples with time strings to be used for loading the h5 files, and a pd.DataFrame with the time-series indexed in the columns.
    
    Arguments:
    
    :zero_time:  -- tuple, e.g. (2012, 07,13,22,15)
    
    :final_time: -- tuple, e.g. (2012, 07,15,00,00)
    
    :time_step: -- zero padded int, the amount of seconds for the timestep. e.g. 300, which means 5 minutes timestep.
    
    '''
    time_series = pd.date_range(start=datetime.datetime(zero_time[0], 
                                                        zero_time[1], 
                                                        zero_time[2], 
                                                        zero_time[3], 
                                                        zero_time[4]), 
                                end=datetime.datetime(final_time[0],
                                                      final_time[1],
                                                      final_time[2],
                                                      final_time[3],
                                                      final_time[4]), 
                                freq='{}S'.format(time_step))
    time_list = []
    for i in time_series:
        time_list += [tuple(i.strftime('%Y,%m,%d,%H,%M').split(','))]
    return time_list, time_series
#%%
def fishnet_loader(a_shp):
    '''Returns a DataFrame containing the WKT polygons of the passed fishnet, and its crs.
    This is, basically, KNMI's 1Km^2 radar grid.
    Arguments

    :a_shp: --string path, the name of the shp file to be read.

    '''
    collection = fiona.open(a_shp, 'r')
    crs = collection.crs
    records = []
    for i in range(len(collection)):
        records.append( next(collection))
    collection.close()
    geoDF = pd.DataFrame({'type': [i['geometry']['type'] for i in records],
                          'properties': [i['properties'] for i in records],
                          'coordinates': [i['geometry']['coordinates'] for i in records]
                          }, index = [i['id'] for i in records])
    geoDF['sequence'] = np.arange(len(geoDF))
    return geoDF, crs
#%%
#def pairwise_fishing_points(a_fishnet, a_rainfall):
#    '''Returns arrays containing shapely polygons representing the passed rainfall_dataframe.
#    
#    Arguments:    
#    
#    :a_fishnet: --pd.DataFrame, created with the fishnet function.
#    
#    :a_ranfall_dataframe: --pd.DataFrame, created with the rainfall_dataframe function.
#
#    '''
#    geo_polygons = a_fishnet.coordinates.apply(lambda x: Polygon(x[0])).values
#    geo_points = a_rainfall.apply(lambda x: Point(x[1], x[0]), axis=1).values    
#    def contains(a_polygon, a_point):
#        return a_polygon.contains(a_point)
#    contains_vectorized = np.vectorize(contains)    
#    fishing_mask = contains_vectorized(geo_polygons, geo_points[:,np.newaxis])#polygons are in columns, points in rows.
#    fished = pd.DataFrame(fishing_mask).stack()[pd.DataFrame(fishing_mask).stack() == True].reset_index(0)
#    fished = fished.merge(a_rainfall, how='left', left_on='level_0', right_on='sequence')
#    fished = fished.merge(a_fishnet, how='left', left_index=True, right_on='sequence', suffixes=('','_grid'))
#    return fished
#%%
def fishing_points(a_fishnet, a_rainfall):
    '''Returns arrays containing shapely polygons representing the passed rainfall_dataframe.
    
    Arguments:    
    
    :a_fishnet: --pd.DataFrame, created with the fishnet function.
    
    :a_ranfall_dataframe: --pd.DataFrame, created with the rainfall_dataframe function.

    '''
    fishnet_index = index.Index()
    geo_polygons = a_fishnet.coordinates.apply(lambda x: Polygon(x[0])).values
    #geo_points = a_rainfall.apply(lambda x: Point(x[1], x[0]), axis=1).values
    for position, cell in enumerate(geo_polygons[:]):
        fishnet_index.insert(position, cell.bounds)
    def intersecter(x):
        return list(fishnet_index.intersection(Point(x[1], x[0]).bounds))
    def get_content(x):
        if x != []:
            return x[0]
        else:
            return -1
    a_rainfall['fishnet_index'] = a_rainfall[a_rainfall.columns[:2]].apply(lambda x: intersecter(x), axis=1).apply(lambda x: get_content(x))
    fished = a_rainfall.merge(a_fishnet, how='left', left_on='fishnet_index', right_on='sequence', suffixes=('','_grid'))
    return fished[fished.fishnet_index != -1]
# <codecell>
def plotter(fished):
    fig = plt.figure(1, figsize = [10,10], dpi = 300)
    ax = fig.add_subplot(111)
    #offset_x = lambda xy: (xy[0] + 0.1, xy[1])
    #offset_y = lambda xy: (xy[0], xy[1] - 0.5)
    for i,j in enumerate(fished.coordinates.apply(lambda x: Polygon(x[0]))):
        ax.add_patch(PolygonPatch(j,alpha=0.1))
        #plt.annotate('polygon {}'.format(i + 1), xy= offset_y(tuple(j.centroid.coords[0])))
    for i,j in enumerate(fished.apply(lambda x: Point(x[1],x[0]), axis=1)):
        ax.add_patch(PolygonPatch(j.buffer(0.07),fc='orange',ec='black'))
        #plt.annotate('point {}'.format(i + 1), xy= offset_x(tuple(j.coords[0])))
    ax.set_xlim(fished.X.min() - 2, fished.X.max() + 2)
    ax.set_ylim(fished.Y.min() - 2, fished.Y.max() + 2)
    ax.set_aspect(1)
    plt.show()
#%%
def wrapper(time_list, time_series, a_fishnet):
    '''Returns pd.DataFrame containing time-parsed rain intensities for the area of interest.
    
    Arguments:    
    
    :a_fishnet: --pd.DataFrame, created with the fishnet function.
    
    :a_ranfall_dataframe: --pd.DataFrame, created with the rainfall_dataframe function.

    '''
    wrap = {}
    for i in time_series:
        wrap[i.to_datetime()] = np.ones(520)#The array has the lenght equals the number of cells in fishnet.
    wrap = pd.DataFrame(wrap)
    for i, j in enumerate(time_list):
        wrap[wrap.columns[i]] = fishing_points(a_fishnet, rainfall_loader(j[0], j[1], j[2], j[3], j[4])).value.values
    fished = fishing_points(a_fishnet, rainfall_loader(j[0], j[1], j[2], j[3], j[4]))
    wrap['coordYX'] =  fished.iloc[:,:2].apply(lambda x: (x[0], x[1]), axis=1)
    return wrap
#%%
def microwave_links_loader(a_timeseries):
    mw_link = pd.read_csv('rainmapmwlink_2012{}.dat'.format(a_timeseries[0].strftime('%m%d%H%M')), header=None, names=[a_timeseries[0]])    
    for i in a_timeseries[1:]:
        mw_link[i] = pd.read_csv('rainmapmwlink_2012{}.dat'.format(i.strftime('%m%d%H%M')), header=None, names=[i])
    return mw_link
#%%
if __name__ == "__main__":
    a_fishnet, a_crs = fishnet_loader('fishnet_KNMI_polygons_1km2_reduced_to_data.shp')    
    time_list, time_series = timer((2012, 07,13,22,15),(2012, 07,15,00,00),900)
    saved_path = os.getcwd()
    os.chdir(os.path.dirname(os.getcwd()))
    os.chdir('data_Adam_MetropolitanSolutions/radar_adam_MS/RAD_NL25_RAC_MFBS_5min')
    wrap = wrapper(time_list, time_series, a_fishnet)
    #a_rainfall = rainfall_loader('2012', '07', '13', '22', '15')
    os.chdir(saved_path)
    #fished = fishing_points(a_fishnet, a_rainfall)
    #plotter(fished)
    os.chdir(os.path.dirname(os.getcwd()))
    os.chdir('data_Adam_MetropolitanSolutions/mwlinkkaart')
    start = datetime.datetime(2012, 7, 13, 22, 15)#first image
    end = datetime.datetime(2012, 7, 15, 0, 0)#last image
    
    
