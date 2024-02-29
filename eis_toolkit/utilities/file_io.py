from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Literal, Sequence, Tuple, Union

from eis_toolkit import exceptions
from eis_toolkit.utilities.checks.raster import check_raster_grids


@beartype
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
    except exceptions.FileReadError:

        # Try to read as a GeoDataFrame
        try:
            data = gpd.read_file(file_path)
        except (ValueError, OSError):

            # Try to read as a DataFrame
            try:
                data = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # If none of the readers succeeded, raise an exception
                raise exceptions.FileReadError(f"Failed to file {file_path} as raster, geodataframe or dataframe")

    return data


@beartype
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
        raise exceptions.FileReadError(f"Failed to read raster data from {file_path}.")
    return data


def read_and_stack_rasters(
    raster_files: Sequence[Path],
    nodata_handling: Literal["convert_to_nan", "unify", "raise_exception", "none"] = "convert_to_nan",
) -> Tuple[np.ndarray, Sequence[rasterio.profiles.Profile]]:
    """
    Read multiple raster files and stack all their bands into a single 3D array.

    Checks that all rasters have the same grid properties. If there are any differences, exception is raised.

    Args:
        raster_files: List of paths to raster files.
        nodata_handling: How to handle raster nodata. convert_to_nan to changes all nodata to np.nan, unify
            changes all rasters to use -9999 as their nodata value, raise_exception raises an exception if
            all rasters do not have the same nodata value, and none does not do anything for nodata.

    Returns:
        3D array with shape (total bands, height, width).
        List of raster profiles.

    Raises:
        NonMatchingRasterMetadataException: If input rasters do not have same grid properties or nodata_handling
            is set to raise exception and mismatching nodata is encountered.
    """
    bands = []
    nodata_values = []
    profiles = []

    for raster_file in raster_files:
        with rasterio.open(raster_file) as raster:
            # Read all bands from each raster
            profile = raster.profile
            for i in range(1, raster.count + 1):
                band_data = raster.read(i)

                if nodata_handling == "convert_to_nan":
                    band_data[band_data == profile["nodata"]] = np.nan
                elif nodata_handling == "unify":
                    band_data[band_data == profile["nodata"]] = -9999
                    profile["nodata"] = -9999
                elif nodata_handling == "raise_exception":
                    nodata_values.append(profile["nodata"])
                    if len(set(nodata_values)) > 1:
                        raise exceptions.NonMatchingRasterMetadataException("Input rasters have varying nodata values.")
                elif nodata_handling == "none":
                    pass
                bands.append(band_data)
            profiles.append(profile)

    if not check_raster_grids(profiles, same_extent=True):
        raise exceptions.NonMatchingRasterMetadataException("Input rasters should have the same properties.")

    # Stack all bands into a single 3D array
    stacked_arrays = np.stack(bands, axis=0)
    return stacked_arrays, profiles


@beartype
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
        raise exceptions.FileReadError(f"Failed to read vector data from {file_path}.")
    return data


@beartype
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
        raise exceptions.FileReadError(f"Failed to read tabular data from {file_path}.")
    return data
