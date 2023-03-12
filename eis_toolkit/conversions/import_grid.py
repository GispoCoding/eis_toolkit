
from typing import List, Tuple
import pandas as pd
import rasterio
from os.path import exists
from eis_toolkit.exceptions import InvalidParameterValueException, NonMatchingImagesExtend, FileReadWriteError, MissingFileOrPath
from eis_toolkit.conversions.raster_to_pandas import *

# *******************************
def _import_grid(
    grids: List[dict]
) -> Tuple[dict, pd.DataFrame, dict]:

    # for every raster-grid
    df = pd.DataFrame()
    fields = {}
    # t (target) first
    q = False
    fl = []
    for dict in grids:                     # !!! prüfen, ob crs, transform height, width gleich
        if not exists(dict['file']):
            raise MissingFileOrPath('file does not extsis: '+dict['file'])
        else:
            try:
                grid = rasterio.open(dict['file'])
                dt = grid.read()[0]   #.T              # add a new columns to the feature-table (dataFrame)
            except:
                raise  FileReadWriteError('file is not readable')
            #nanv = grid.meta['nodata']
            #da = grid.read()
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
                    fl.append('height and/or width differs in the imported grids ') 
                if meta['crs'] != grid.meta['crs']:
                    fl.append('crs differs in the imported grids ')
                if meta['transform'] != grid.meta['transform']:
                    fl.append('extend/cellsize differs in the imported grids ')  
                meta = grid.meta
            fields[dict['name']]=dict['type']
    if len(fl) > 0:
        raise NonMatchingImagesExtend(fl[0])
    # remove the file-name out of the fields-dictionaries (not nessesary)
    # fields = grids
    # for gr in fields:
    #     del gr["file"]

    return fields, df, meta

# *******************************
def import_grid(  # type: ignore[no-any-unimported]
    grids: List[dict]
) -> Tuple[dict, pd.DataFrame, dict]: 

    """
        Add a list of rasters (grids) as columns to new pandas DataFrame.
            import_grid reads all rasterformats of Python-Modu "rasterio": 
            e.g. geoTiff (tif), tif with tfw-file(ESRI), ESRI-Grid (no extension in the filename)... .
        Write the "name" and the "type" of each of this columns to a new dictionary "fields" 
        All rasterfiles should have the same crs (coordinates) as well as the same width and height and cellsize.
    Args:
        grids (List of dictionaries): containing 
            "name" a unique name for each grid, 
            "file" the filename of each grid and
            "type" the type of each grid (v - value, c - categorised, b - boolean, t - target)
                In case of reading input raster for prediction, no 't' - grids are needed.
    Returns:
        - dictionary:  name, type and nodatavalue of each column
        - pandas DataFrame: One dataframe column of each imported grids
        - dictionary:  metadata of the first imported grid 
                     containing the keys: driver, dtype, nodata, width, height, count (=1), crs, transform
    """

    # Argument evaluation
    fl = []
    if not (isinstance(grids, list)):
        fl.append('argument lists is not a list')   
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])
    if len(grids) == 0:
        raise InvalidParameterValueException ('Argunment grids is empty')
    if not (grids[0].__class__.__name__ == 'dict'):         #(isinstance(grids[0],dict)):
        raise InvalidParameterValueException ('The grids list contains no dictionaries')

    # v,b,c-fields at least 1, t no more then 1

    if len(list(counter for counter, fld in enumerate(grids) if fld['type'] in ['v','c','b'])) < 1:
        raise InvalidParameterValueException ('There are no v-, c- or b-fields in fields argument') 
    if len(list(counter for counter, fld in enumerate(grids) if fld['type'] in ['t'])) > 1:
        raise InvalidParameterValueException ('There are more then one t-fields in fields argument')  

    fields, data_frame, meta = _import_grid( 
        grids = grids
    )                       #, add_img_coord = add_img_coord, height = height, width = width)

    return fields, data_frame, meta   #, columns, cats

