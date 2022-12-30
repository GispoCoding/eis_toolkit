"""
State an Dezember 15 2022
@author: torchala 
""" 

## zu Verzeichnis conversions
# Stand: in Bearbeitung
#   Verbessern:  - die Metadaten mit in die fields-list als dictionary in jedem dictionary
#                - bands?  erstmal nicht
#                - prüfen, ob alle grids die gleiche cellsize, Koordinaten und with and height haben
#                - prüfen, ob die Namen eindeutig sind (alle unterschiedlich. )

from typing import List, Tuple
import pandas as pd
import rasterio
from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.conversions.raster_to_pandas import *

# *******************************
def _import_grid(
    grids: List[dict]
) -> Tuple[pd.DataFrame,dict, list]:

    # for every raster-grid
    df = pd.DataFrame()
    fields = {}
    metalist = []
    for dict in grids:
        grid = rasterio.open(dict['file'])
        #da = grid.read()
        dt = grid.read()[0] #.T              # add a new columns to the feature-table (dataFrame)
        dtrans = np.ravel(dt)
        df[dict['name']] = pd.DataFrame(dtrans) #, columns=...)        #Xvdf.join(Xcdf)
        metalist.append(grid.meta.copy())           # add a new item of metadatada dictionaries to the list
        fields[dict['name']]=dict['type']
    # remove the file-name out of the fileds-dictionaries (not nessesary)
    # fields = grids
    # for gr in fields:
    #     del gr["file"]

    return df,fields, metalist

# *******************************
def import_grid(  # type: ignore[no-any-unimported]
    grids: List[dict]
) -> Tuple[pd.DataFrame,dict, list]: 

    """
    add a list of rasters (grids) as columns to pandas DataFrame.
    write the "name" and the "type" of each of this columns to a dictionary "fields"

    Args:
        grids (List of dictionaries): containing 
            "name" a unique name for each grid, 
            "file" the filename of each grid and
            "type" the type of each grid (v - value, c - categorised, b - boolean, t - target)

    Returns:
        pandas DataFrame: one pandas dataframe of alle imported grids 
        dictionary:  name, type and nodatavalue of each column
        List of dicts:  metadata (dictionary) of the each imported grid 
    """

    data_frame, fields, metalist = _import_grid( 
        grids = grids
    ) #, add_img_coord = add_img_coord, height = height, width = width)

    return data_frame, fields, metalist   #, columns, cats

