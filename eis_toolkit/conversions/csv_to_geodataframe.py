import csv as reader
from pathlib import Path

import geopandas
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.exceptions import (
    InvalidColumnIndexException,
    InvalidParameterValueException,
    InvalidWktFormatException,
)


def _csv_to_geodataframe(
    csv: Path,
    indexes: Sequence[int],
    target_crs: int,
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
                geodataframe = geopandas.GeoDataFrame(df, crs=target_crs, geometry=geoms)
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
                    df, crs=target_crs, geometry=geopandas.points_from_xy(df[geom_x], df[geom_y])
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
                geodataframe = geopandas.GeoDataFrame(df, crs=target_crs, geometry=geoms)
                return geodataframe
            except:  # noqa: E722
                raise InvalidWktFormatException
        else:
            if len(df.columns) < indexes[0] or len(df.columns) < indexes[1]:
                raise InvalidColumnIndexException
            try:
                geodataframe = geopandas.GeoDataFrame(
                    df, crs=target_crs, geometry=geopandas.points_from_xy(df[indexes[0]], df[indexes[1]])
                )
                return geodataframe
            except:  # noqa: E722
                raise InvalidParameterValueException


@beartype
def csv_to_geodataframe(
    csv: Path,
    indexes: Sequence[int],
    target_crs: int,
) -> geopandas.GeoDataFrame:
    """
    Read CSV file to a GeoDataFrame.

    Usage of single index expects valid WKT geometry.
    Usage of two indexes expects POINT feature(s) X-coordinate as the first index and Y-coordinate as the second index.

    Args:
        csv: Path to the .csv file to be read.
        indexes: Index(es) of the geometry column(s).
        target_crs: Target CRS as an EPSG code.

    Returns:
        CSV file read to a GeoDataFrame.

    Raises:
        InvalidColumnIndexException: There is a mismatch between the provided indexes and the shape of
            the dataframe read from the csv.
        InvalidParameterValueException: Unable to create a GeoDataFrame with point features from the given input
            parameters.
        InvalidWktFormatException: Unable to create a GeoDataFrame of WKT geometry from the given input parameters.
    """

    data_frame = _csv_to_geodataframe(
        csv=csv,
        indexes=indexes,
        target_crs=target_crs,
    )
    return data_frame
