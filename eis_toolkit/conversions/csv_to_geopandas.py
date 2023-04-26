import csv as reader
from pathlib import Path
from typing import List

import geopandas
import pandas as pd

from eis_toolkit.exceptions import (
    InvalidColumnIndexException,
    InvalidParameterValueException,
    InvalidWktFormatException,
)


def _csv_to_geopandas(  # type: ignore[no-any-unimported]
    csv: Path,
    indexes: List[int],
    target_EPSG: int,
) -> geopandas.GeoDataFrame:

    with csv.open(mode="r") as f:
        has_header = reader.Sniffer().has_header(f.read(1024))

    if has_header:
        df = pd.read_csv(csv)
        if len(indexes) == 1:
            if len(df.columns) < indexes[0]:
                raise InvalidColumnIndexException
            column_names = []
            for row in df:
                column_names.append(row)
            geom_column = column_names[indexes[0]]
            try:
                geoms = geopandas.GeoSeries.from_wkt(df[geom_column])
                geodataframe = geopandas.GeoDataFrame(df, crs=target_EPSG, geometry=geoms)
                return geodataframe
            except:  # noqa: E722
                raise InvalidWktFormatException

        else:
            if len(df.columns) < indexes[0] or len(df.columns) < indexes[1]:
                raise InvalidColumnIndexException
            column_names = []
            for row in df:
                column_names.append(row)
            try:
                geom_x = column_names[indexes[0]]
                geom_y = column_names[indexes[1]]
                geodataframe = geopandas.GeoDataFrame(
                    df, crs=target_EPSG, geometry=geopandas.points_from_xy(df[geom_x], df[geom_y])
                )
                return geodataframe
            except:  # noqa: E722
                raise InvalidParameterValueException
    else:
        df = pd.read_csv(csv, header=None)
        if len(indexes) == 1:
            if len(df.columns) < indexes[0]:
                raise InvalidColumnIndexException
            try:
                geoms = geopandas.GeoSeries.from_wkt(df[indexes[0]])
                geodataframe = geopandas.GeoDataFrame(df, crs=target_EPSG, geometry=geoms)
                return geodataframe
            except:  # noqa: E722
                raise InvalidWktFormatException
        else:
            if len(df.columns) < indexes[0] or len(df.columns) < indexes[1]:
                raise InvalidColumnIndexException
            try:
                geodataframe = geopandas.GeoDataFrame(
                    df, crs=target_EPSG, geometry=geopandas.points_from_xy(df[indexes[0]], df[indexes[1]])
                )
                return geodataframe
            except:  # noqa: E722
                raise InvalidParameterValueException


def csv_to_geopandas(  # type: ignore[no-any-unimported]
    csv: Path,
    indexes: List[int],
    target_EPSG: int,
) -> geopandas.GeoDataFrame:
    """
    Convert CSV file to geopandas DataFrame.

    Usage of single index expects valid WKT geometry.
    Usage of two indexes expects POINT feature(s) X-coordinate as the first index and Y-coordinate as the second index.

    Args:
        csv: path to the .csv file to be converted.
        indexes: index(es) of the geometry column(s).
        target_EPSG: Target crs as EPSG code.

    Returns:
        geodataframe: csv converted to geopandas geodataframe.
    """

    data_frame = _csv_to_geopandas(
        csv=csv,
        indexes=indexes,
        target_EPSG=target_EPSG,
    )
    return data_frame
