import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Any, Literal, Sequence, Tuple, Union

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


def get_output_paths_from_inputs(
    input_paths: Sequence[Path], directory: Path, suffix: str, extension: str
) -> Sequence[Path]:
    """
    Get output paths using input paths to extract file name bases.

    Combines directory, file name extracted from input path, suffix and extension.
    Include dot in the extension, for example '.tif'.

    This tool is designed mainly for convenience in CLI functions.

    Args:
        input_paths: Input paths.
        directory: Path of the output directory.
        suffix: Common suffix added to the end of each output file name, for example "nodata_unified".
        extension: The extension used for the output path, for example ".tif".

    Returns:
        List of output paths.
    """
    output_paths = []
    for input_path in input_paths:
        input_file_name_with_extension = os.path.split(input_path)[1]
        input_file_name = os.path.splitext(input_file_name_with_extension)[0]
        output_file_name = f"{input_file_name}_{suffix}"
        output_path = directory.joinpath(output_file_name + extension)
        output_paths.append(output_path)

    return output_paths


def get_output_paths_from_names(
    file_names: Sequence[str], directory: Path, suffix: str, extension: str
) -> Sequence[Path]:
    """
    Get output paths directly from given file names.

    Combines directory, file name, suffix and extension.
    Include dot in the extension, for example '.tif'.

    This tool is designed mainly for convenience in CLI functions.

    Args:
        input_paths: Raw file names.
        directory: Path of the output directory.
        suffix: Common suffix added to the end of each output file name, for example "nodata_unified".
        extension: The extension used for the output path, for example ".tif".

    Returns:
        List of output paths.
    """
    output_paths = []
    for name in file_names:
        output_file_name = f"{name}_{suffix}"
        output_path = directory.joinpath(output_file_name + extension)
        output_paths.append(output_path)

    return output_paths


def get_output_paths_from_common_name(
    outputs: Sequence[Any], directory: Path, common_name: str, extension: str, first_num: int = 1
) -> Sequence[Path]:
    """
    Get output paths for cases where outputs should be just numbered.

    Combines directory, given common file name, number and extension. Outputs are used
    to get the number used as suffix.
    Include dot in the extension, for example '.tif'.

    This tool is designed mainly for convenience in CLI functions.

    Args:
        input_paths: Outputs. Used just to iterate and get numbers for suffixes.
        directory: Path of the output directory.
        common_name: Common name used as the basis of each output file name. A number is appended to this.
        extension: The extension used for the output path, for example ".tif".
        first_num: The first number used as a suffix.

    Returns:
        List of output paths.
    """
    output_paths = []
    for i in range(first_num, len(outputs) + first_num):
        output_path = directory.joinpath(common_name + f"_{i}" + extension)
        output_paths.append(output_path)

        output_paths.append

    return output_paths
