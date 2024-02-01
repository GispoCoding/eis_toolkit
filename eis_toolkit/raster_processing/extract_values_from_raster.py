import os

import geopandas as gpd
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import NonMatchingParameterLengthsException


def _extract_values_from_raster(
    raster_list: Sequence[rasterio.io.DatasetReader],
    geodataframe: gpd.GeoDataFrame,
    raster_column_names: Optional[Sequence[str]],
) -> pd.DataFrame:

    data_frame = pd.DataFrame()

    points = geodataframe["geometry"].apply(lambda point: (point.xy[0][0], point.xy[1][0]))

    data_frame["x"] = points.apply(lambda point: (point[0]))
    data_frame["y"] = points.apply(lambda point: (point[1]))

    for i, raster in enumerate(raster_list):
        raster_values = [value for value in raster.sample(zip(data_frame["x"], data_frame["y"]))]

        for band_number in range(raster.count):
            if raster_column_names is not None:
                if raster.count > 1:
                    band_column_name = str(raster_column_names[i]) + "_" + str(band_number + 1)
                else:
                    band_column_name = str(raster_column_names[i])
            else:
                if raster.count > 1:
                    band_column_name = os.path.splitext(raster.name)[0].rsplit("/", 1)[-1] + "_" + str(band_number + 1)
                else:
                    band_column_name = os.path.splitext(raster.name)[0].rsplit("/", 1)[-1]
            data_frame[band_column_name] = [array[band_number] for array in raster_values]

    return data_frame


@beartype
def extract_values_from_raster(
    raster_list: Sequence[rasterio.io.DatasetReader],
    geodataframe: gpd.GeoDataFrame,
    raster_column_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Extract raster values using point data to a DataFrame.

       If custom column names are not given, column names are file_name for singleband files
       and file_name_bandnumber for multiband files. If custom column names are given, there
       should be column names for each raster provided in the raster list.

    Args:
        raster_list: List to extract values from.
        geodataframe: Object to extract values with.
        raster_column_names: List of optional column names for bands.

    Returns:
        Dataframe with x & y coordinates and the values from the raster file(s) as columns.

    Raises:
        NonMatchingParameterLengthsException: raster_list and raster_columns_names have different lengths.
    """
    if raster_column_names == []:
        raster_column_names = None

    if raster_column_names is not None and len(raster_list) != len(raster_column_names):
        raise NonMatchingParameterLengthsException("Raster list and raster columns names have different lengths.")

    data_frame = _extract_values_from_raster(
        raster_list=raster_list, geodataframe=geodataframe, raster_column_names=raster_column_names
    )

    return data_frame
