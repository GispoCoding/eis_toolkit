from os.path import exists

import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import List, Tuple

from eis_toolkit.exceptions import FileReadError, InvalidParameterValueException, MatchingRasterGridException


# *******************************
@beartype
def _import_grid(grids: List[dict]) -> Tuple[dict, pd.DataFrame, dict]:

    # for every raster-grid
    df = pd.DataFrame()
    fields = {}

    q = False
    for dict in grids:  # check if crs, transform, height and width are equal for the grids
        if not exists(dict["file"]):
            raise FileReadError("file does not extsis: " + dict["file"])
        else:
            try:
                grid = rasterio.open(dict["file"])
                dt = grid.read()[0]             # add a new columns to the feature-table (dataFrame)
            except:
                raise FileReadError("file is not readable")
            dtrans = np.ravel(dt)
            df_nd = pd.DataFrame(dtrans)
            df_nd = df_nd.replace({grid.meta["nodata"]: np.nan})
            df[dict["name"]] = df_nd
            if q == False:
                meta = grid.meta  # driver, dtype, nodata, width, height, count (=1), crs, transform
                q = True
            else:
                if meta["height"] != grid.meta["height"] or meta["width"] != grid.meta["width"]:
                    raise MatchingRasterGridException("height and/or width differs in the imported grids ")
                if meta["crs"] != grid.meta["crs"]:
                    raise MatchingRasterGridException("crs differs in the imported grids ")
                if meta["transform"] != grid.meta["transform"]:
                    raise MatchingRasterGridException("extend/cellsize differs in the imported grids ")
                meta = grid.meta
            fields[dict["name"]] = dict["type"]

    return fields, df, meta


# *******************************
@beartype
def import_grid(grids: List[dict]) -> Tuple[dict, pd.DataFrame, dict]:

    """
        Add a list of rasters (grids) as columns to new pandas DataFrame.
            import_grid reads all rasterformats of Python-Modu "rasterio":
            e.g. geoTiff (tif), tif with tfw-file(ESRI), ESRI-Grid (no extension in the filename)... .
        Write the "name" and the "type" of each of this columns to a new dictionary "fields"
        All rasterfiles should have the same crs (coordinates) as well as the same width and height and cellsize.
    Args:
        grids: containing
            "name" a unique name for each grid,
            "file" the filename of each grid and
            "type" the type of each grid (v - value, c - categorised, b - boolean, t - target)
                In case of reading input raster for prediction, no 't' - grids are needed.
    Returns:
        - dictionary:  name, type and nodatavalue of each column
        - pandas DataFrame: One column of each imported grids
        - dictionary: metadata of the first imported grid
                     containing the keys: driver, dtype, nodata, width, height, count (=1), crs, transform
    """

    # Argument evaluation
    if len(grids) == 0:
        raise InvalidParameterValueException("Argunment is an empty list")
    # v,b,c-fields at least 1, t no more then 1
    if len(list(counter for counter, fld in enumerate(grids) if fld["type"] in ["v", "c", "b"])) < 1:
        raise InvalidParameterValueException("There are no v-, c- or b-fields in fields argument")
    if len(list(counter for counter, fld in enumerate(grids) if fld["type"] in ["t"])) > 1:
        raise InvalidParameterValueException("There are more then one t-fields in fields argument")

    fields, data_frame, meta = _import_grid(
        grids=grids
    )

    return fields, data_frame, meta
