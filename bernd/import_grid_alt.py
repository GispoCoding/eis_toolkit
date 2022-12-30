"""
State an November 30 2022
@author: torchala 
""" 
# Stand: getestet mit eingen Beispiel-Bidern und mehreren Bändern:   soweit fertig
# - function raster_to_pandas will be used to import one raster(grid)
# verbessungspotenzial / offene Fragen: 
#    - Für das mehrfache Aufrufen ein übergeordnets Modul erforderlich? (siehe unten)
#    - weitere Tests, z. B. andere Bildformate: ESRI-Grid, BigTiff...?
#    - column-names are unique? If not: extend the name: name_1 or _2... 

    ### to check if: 
    #  - import file is a grid for import i rasterio
    #  - bands is a valid number of bands
    #  - name is unique in pandas columns
    #  - height and width have the same number as the first grid
    #  - categ: integer >=0, if > 0: grid must be integer with values <= categ (import as pandas: just now)
    #   if categ >0: just one band will be uses !!!
 
    # if bands is not None:
    #     if not isinstance(bands, list):
    #         raise InvalidParameterValueException
    #     elif not all(isinstance(band, int) for band in bands):
    #         raise InvalidParameterValueException

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show
from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.conversions.raster_to_pandas import *

# *******************************
def _import_grid(
    grids: dict,      # list of the name of the name of the 


    grid: rasterio.DatasetReader,
    name: str,
    df: Optional[pd.DataFrame] = None,
    categ: Optional[int] = 0,
    bands: Optional[List[int]] = None
) -> Tuple[pd.DataFrame,dict]:

    grid = rasterio.open(grid)  ####
    # grid = grid.read()
    # plt.imshow(np.squeeze(grid)) ######
    out_meta = grid.meta.copy()

    tmp = raster_to_pandas(raster = grid, bands = bands)   # , add_img_coord = add_img_coord)
    
    ### OneHotEncoder (categ) and new colums names
    if categ > 0:
        
        if not pd.api.types.is_integer_dtype(tmp.dtypes[0]):               # Integer-Type
            raise InvalidParameterValueException ('***  no integer type for categories: ' + grid.name) 
        if tmp.min()[0] < 0:                                               # values < 0
             raise InvalidParameterValueException ('***  less 0 categories: ' + grid.name)                
        if tmp.max()[0] > categ:                                           # vanues (0,1,....) of categ ()
            raise InvalidParameterValueException ('***  integer categories larger then parameter: ' + grid.name)

        if len(tmp.columns) == 1:
            tmp.rename(columns={"band_1": name}, inplace = True)

        from sklearn.preprocessing import OneHotEncoder          # see www: sklearn OneHotEncode
        enc = OneHotEncoder (handle_unknown='ignore', sparse = False)
        enc.fit(tmp)
        tmp = enc.transform(tmp)
        tmp = pd.DataFrame(tmp, columns = enc.get_feature_names_out([].append(name)))
    else:     # ist not 'catgorie', then name"band_..

        if len(tmp.columns) == 1:
            tmp.rename(columns={"band_1": name}, inplace = True)
        else:
            d = {}
            for tc in tmp.columns:
                d[tc] = name+'_'+tc
            tmp.rename(columns=d,inplace = True)

    if df is None:      # add image as a new column to the dataframe 
        df = tmp
    else:
        # df = df.concat([df,tmp], axis = 1)
        df = df.join(tmp)

    return df, out_meta

# *******************************
def import_grid(  # type: ignore[no-any-unimported]
    grid: rasterio.DatasetReader,
    name: str,
    df: Optional[pd.DataFrame] = None,
    categ: Optional[int] = 0, 
    bands: Optional[List[int]] = None

) -> Tuple[pd.DataFrame,dict]:

    """
    add raster (grid) as furthe column to pandas DataFrame.

    - If Dataframe df not exists, DataFrame will be created (one column): 
      Name is the column name, to use (! name shold be unique)
    - categ > 0: grid is a grid with 0-n categories (classes). i.g. landuse , geological category ....
    - If bands are not given, all bands of the image are used for conversion. 
      Selected bands are named based on their index 
      band_1, band_2,...,band_n. 

    Args:
        grid: to imported rasterfile (e.g. GeoTiff, ESRI-GRID...)
        name (str): Name to be used as the column-name of the new column 
            (in case of more than 1 band tha name will be extenden by _band_1...)
        df (Pandas DataFrame): existing DataFrame to be extended with an new column 
           df get new columns from the bands of the grid.
           if df not exists a new one will be ceated
        categ (int): Number of Classes, if categ > 0 just one band will be used (no multiband images in this case)
        raster (rasterio.DatasetReader): Raster (e.g. filename) to be converted.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from 1. Defaults to None. 

    Returns:
        pd.DataFrame: Raster converted to new columns of pandas dataframe
    """

    data_frame, out_meta = _import_grid( grid = grid, 
        name = name, 
        df = df, 
        categ = categ, 
        bands = bands
    ) #, add_img_coord = add_img_coord, height = height, width = width)

    return data_frame, out_meta


