import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Iterable


def _extract_values_from_raster(  # type: ignore[no-any-unimported]
    raster_list: Iterable[rasterio.io.DatasetReader],
    geodataframe: gpd.GeoDataFrame,
    raster_column_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:

    data_frame = pd.DataFrame()

    points = geodataframe["geometry"].apply(lambda point: (point.xy[0][0], point.xy[1][0]))

    data_frame["x"] = points.apply(lambda point: (point[0]))
    data_frame["y"] = points.apply(lambda point: (point[1]))

    for raster_number in range(len(raster_list)):

        raster = raster_list[raster_number]

        raster_values = [value for value in raster.sample(zip(data_frame["x"], data_frame["y"]))]

        band_column_name = ""
        for band_number in range(raster.count):
            if raster_column_names is not None:
                if raster.count > 1:
                    band_column_name = str(raster_column_names[raster_number]) + "_" + str(band_number + 1)
                else:
                    band_column_name = str(raster_column_names[raster_number])
            else:
                if raster.count > 1:
                    band_column_name = os.path.splitext(raster.name)[0].rsplit("/", 1)[-1] + "_" + str(band_number + 1)
                else:
                    band_column_name = os.path.splitext(raster.name)[0].rsplit("/", 1)[-1]
            data_frame[band_column_name] = [array[band_number] for array in raster_values]

        replaceable_values = {-999.999: np.NaN, -999999.0: np.NaN}
        data_frame = data_frame.replace({band_column_name: replaceable_values})

    return data_frame


@beartype
def extract_values_from_raster(  # type: ignore[no-any-unimported]
    raster_list: Iterable[rasterio.io.DatasetReader],
    geodataframe: gpd.GeoDataFrame,
    raster_column_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Extract raster values using point data to a dataframe.

       If custom column names are not given, column names are file_name for singleband files
       and file_name_bandnumber for multiband files.

    Args:
        raster_list: list to extract values from.
        geodataframe: object to extract values with.
        raster_column_names: list of optional column names for bands.

    Returns:
        Dataframe with x & y coordinates and the values from the raster file(s) as columns.
    """
    if raster_column_names == []:
        raster_column_names = None

    data_frame = _extract_values_from_raster(
        raster_list=raster_list, geodataframe=geodataframe, raster_column_names=raster_column_names
    )

    return data_frame
