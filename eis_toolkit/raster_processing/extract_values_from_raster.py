from typing import List, Optional

import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
import sys

#from eis_toolkit.exceptions import InvalidParameterValueException


def _extract_values_from_raster( # type: ignore[no-any-unimported]
    raster_list: List[rasterio.io.DatasetReader], 
    shapefile: gpd.GeoDataFrame,
    raster_column_names: Optional[str] = None
) -> pd.DataFrame:
    
    data_frame = pd.DataFrame()
    
    for raster_number in range(len(raster_list)):

        points = shapefile['geometry'].apply(lambda point: (point.xy[0][0], point.xy[1][0]))

        data_frame['x'] = points.apply(lambda point: (point[0]))
        data_frame['y'] = points.apply(lambda point: (point[1]))

        coord_list = [(x,y) for x,y in points]
        raster_values = [value for value in raster.sample(coord_list)] 

        column = ""        
        raster = raster_list[raster_number]
        for band_number in range(raster.count):
            if raster_column_names is not None:
                if raster.count > 1:
                    column = str(raster_column_names[raster_number]) + "_" + str(band_number+1)
                else:
                    column = str(raster_column_names[raster_number])
            else:
                if raster.count > 1:
                    column = os.path.splitext(raster.name)[0].rsplit('/',1)[-1] + "_" + str(band_number+1)
                else:
                    column = os.path.splitext(raster.name)[0].rsplit('/',1)[-1]
            data_frame[column] = [array[band_number] for array in raster_values] 

        data_frame[column] = data_frame[column].replace(-999999.0, np.NaN)
    return data_frame

def extract_values_from_raster( # type: ignore[no-any-unimported]
    raster_list: List[rasterio.io.DatasetReader], 
    shapefile: gpd.GeoDataFrame,
    raster_column_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Extracts raster values using point data to a dataframe.
       
       If custom column names are not given, column names are file_name for singleband files and file_name_bandnumber for multiband files.
        
    Args:
        raster_list (List[rasterio.io.DatasetReader]): list to extract values from.
        shapefile (geopandas.GeoDataFrame): object to extract values with.
        raster_column_names (List[str]): list of optional column names for bands.
        
    Returns:
        pandas.DataFrame: Dataframe with x & y coordinates and the values from the raster file(s) as columns.
    """
    if raster_column_names is not None:
        if not isinstance(raster_column_names, list):
            raise InvalidParameterValueException
        elif not all(isinstance(raster_column_name, str) for raster_column_name in raster_column_names):
            raise InvalidParameterValueException
            
    if not isinstance(raster_list, list):
        raise InvalidParameterValueException
    elif not all(isinstance(raster, rasterio.io.DatasetReader) for raster in raster_list):
        raise InvalidParameterValueException
            
    data_frame = _extract_values_from_raster(
        raster_list = raster_list, 
        shapefile = shapefile, 
        raster_column_names = raster_column_names)
    
    return data_frame