import rasterio
import os
import numpy as np
import pandas as pd
import geopandas as gpd

def _raster_conversion_to_dataframe(
    raster_list: list[rasterio.io.DatasetReader], 
    shapefile: gpd.GeoDataFrame,
    raster_column_names = None
) -> pd.DataFrame:
    
    dataframe = pd.DataFrame()
    
    for raster_number in range(len(raster_list)):
        raster = raster_list[raster_number]
        #if raster.count > 1:
            #dataframe = read_multiband_data(dataframe,raster,shapefile,raster_column_names,raster_number)
        #else:
        dataframe = read_raster_data(dataframe,raster,shapefile,raster_column_names,raster_number)
    return dataframe

def raster_conversion_to_dataframe(
    raster_list: list[rasterio.io.DatasetReader], 
    shapefile: gpd.GeoDataFrame,
    raster_column_names = None
) -> pd.DataFrame:
    """Extracts raster values using point data to a dataframe.

    Args:
        raster_list list[rasterio.io.DatasetReader]: list to extract values from.
        shapefile (geopandas.GeoDataFrame): object to extract values with.
        raster_column_names list[]: list of optional column names for bands.
        
    Returns:
        dataframe (pandas.DataFrame): Dataframe with x & coordinates and the values from the raster file as columns.
    """
    
    dataframe = _raster_conversion_to_dataframe(raster_list, shapefile, raster_column_names)
    
    return dataframe

def read_raster_data(dataframe,raster,shapefile,
    raster_column_names = None,raster_number = None):

    points = shapefile['geometry'].apply(lambda point: (point.xy[0][0], point.xy[1][0]))
    
    dataframe['x'] = points.apply(lambda point: (point[0]))
    dataframe['y'] = points.apply(lambda point: (point[1]))
    
    coord_list = [(x,y) for x,y in points]
    raster_values = [value for value in raster.sample(coord_list)] 
    
    column = ""
    for i in range(raster.count):
        if raster_column_names is not None:
            if raster.count > 1:
                column = str(raster_column_names[raster_number]) + "_" + str(i+1)
            else:
                column = str(raster_column_names[raster_number])
        else:
            if raster.count > 1:
                column = os.path.splitext(raster.name)[0].rsplit('/',1)[-1] + "_" + str(i+1)
            else:
                column = os.path.splitext(raster.name)[0].rsplit('/',1)[-1]
        dataframe[column] = [array[i] for array in raster_values] 

        # -999999.0 values are outside the region.
        dataframe[column] = dataframe[column].replace(-999999.0, np.NaN)
    
    return dataframe