
from typing import List, Tuple
import pandas as pd
import rasterio
from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.conversions.raster_to_pandas import *

# *******************************
def _import_grid(
    grids: List[dict]
) -> Tuple[pd.DataFrame,dict,dict]:

    if len(grids) == 0:
        raise InvalidParameterValueException ('***  grids is empty')
        
    # for every raster-grid
    df = pd.DataFrame()
    fields = {}
    # t (target) first
    q = False
    for dict in grids:                     # !!! prüfen, ob crs, transform height, width gleich
        grid = rasterio.open(dict['file'])
        #nanv = grid.meta['nodata']
        #da = grid.read()
        dt = grid.read()[0] #.T              # add a new columns to the feature-table (dataFrame)
        dtrans = np.ravel(dt)
        #df['y'] = df['y'].replace({grid.meta['nodata']: np.nan})
        df_nd = pd.DataFrame(dtrans)
        df_nd = df_nd.replace({grid.meta['nodata']: np.nan})
        df[dict['name']] = df_nd        #pd.DataFrame(dtrans) #, columns=...)        #Xvdf.join(Xcdf)
        #metalist.append(grid.meta.copy())           # add a new item of metadatada dictionaries to the list
        if q == False:
            meta = grid.meta         #driver, dtype, nodata, width, height, count (=1), crs, transform
            q = True
        else:
            if meta['height'] != grid.meta['height'] or meta['width'] != grid.meta['width']:
                raise InvalidParameterValueException ('*** height and/or width differs in the imported grids ') 
            meta = grid.meta
        fields[dict['name']]=dict['type']
    # remove the file-name out of the fields-dictionaries (not nessesary)
    # fields = grids
    # for gr in fields:
    #     del gr["file"]

    return df,fields,meta

# *******************************
def import_grid(  # type: ignore[no-any-unimported]
    grids: List[dict]
) -> Tuple[pd.DataFrame,dict,dict]: 

    """
        Add a list of rasters (grids) as columns to pandas DataFrame.
        import_grid reads all rasterformats of "rasterio" (Pthon-Model): 
        e.g. geoTiff (tif), tif with tfw-file(ESRI), ESRI-Grid (no extension in the filename).
        All rasterfles should have the same crs (coordinates) as well as width and height
        Write the "name" and the "type" of each of this columns to a dictionary "fields"
    Args:
        grids (List of dictionaries): containing 
            "name" a unique name for each grid, 
            "file" the filename of each grid and
            "type" the type of each grid (v - value, c - categorised, b - boolean, t - target)
    Returns:
        pandas DataFrame: one pandas dataframe of alle imported grids 
        dictionary:  name, type and nodatavalue of each column
        dictionary:  metadata of the first imported grid 
                    containing the keys: driver, dtype, nodata, width, height, count (=1), crs, transform
    """

    data_frame,fields,meta = _import_grid( 
        grids = grids
    )                       #, add_img_coord = add_img_coord, height = height, width = width)

    return data_frame,fields,meta   #, columns, cats
