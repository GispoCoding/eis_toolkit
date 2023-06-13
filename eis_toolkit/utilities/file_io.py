from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import rasterio

from eis_toolkit.exceptions import FileReadError


def read_file(file_path: Path) -> Union[rasterio.io.DatasetReader, gpd.GeoDataFrame, pd.DataFrame]:
    """Read an input file trying different readers.
    First tries to read to a rasterio DatasetReader, then to a GeoDataFrame, then to a DataFrame.
    If none of the readers succeed, raises an exception.
    Args:
        file_path: Input file path.
    Returns:
        The input file data in the opened format.
    Raises:
        FileReadError:None of the readers succeeded to read the input file.
    """

    # Try to read as a raster first
    try:
        data = read_raster(file_path)
    except FileReadError:

        # Try to read as a GeoDataFrame
        try:
            data = gpd.read_file(file_path)
        except (ValueError, OSError):

            # Try to read as a DataFrame
            try:
                data = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # If none of the readers succeeded, raise an exception
                raise FileReadError(f"Failed to file {file_path} as raster, geodataframe or dataframe")

    return data


def read_raster(file_path: Path) -> rasterio.io.DatasetReader:
    """Read a raster file to a rasterio DatasetReader.
    Args:
        file_path: Input file path.
    Returns:
        File data as a Rasterio DatasetReader.
    Raises:
        FileReadError: Rasterio failed to open the input file.
    """
    try:
        data = rasterio.open(file_path)
    except rasterio.errors.RasterioIOError:
        raise FileReadError(f"Failed to read raster data from {file_path}.")
    return data


def read_vector(file_path: Path) -> gpd.GeoDataFrame:
    """Read a vector file to a GeoDataFrame.
    Args:
        file_path: Input file path.
    Returns:
        File data as a GeoDataFrame.
    Raises:
        FileReadError: Geopandas failed to read the input file.
    """
    try:
        data = gpd.read_file(file_path)
    except (ValueError, OSError):
        raise FileReadError(f"Failed to read vector data from {file_path}.")
    return data


def read_tabular(file_path: Path) -> pd.DataFrame:
    """Read tabular data to a DataFrame.
    Args:
        file_path: Input file path.
    Returns:
        File data as a DataFrame.
    Raises:
        FileReadError: Pandas failed to open the input file.
    """
    try:
        data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        raise FileReadError(f"Failed to read tabular data from {file_path}.")
    return data
