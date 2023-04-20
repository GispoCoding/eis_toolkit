from pathlib import Path
from typing import List, Optional

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException, UnsupportedFileExtensionException


def get_supported_raster_extensions() -> List[str]:
    """Get raster file extensions supported by rasterio."""
    extensions = list(rasterio.drivers.raster_driver_extensions().keys())
    return extensions


def get_supported_vector_extensions() -> List[str]:
    """Get vector file extensions supported."""
    extensions = ["shp", "geojson", "gpkg", "csv", "kml", "gml"]
    return extensions


def read_spatial(file_path: Path) -> rasterio.io.DatasetReader | gpd.GeoDataFrame:
    """Read and open a vector or raster file."""
    file_extension = file_path.split(".")[-1]
    if file_extension in get_supported_vector_extensions:
        data = rasterio.open(file_path)
    elif file_extension in get_supported_vector_extensions():
        data = gpd.read_file(file_path)
    else:
        raise UnsupportedFileExtensionException(
            f"File extension {file_extension} is not supported. Provide a raster or vector file."
        )
    return data


def read_raster(file_path: Path) -> rasterio.io.DatasetReader:
    """Read and open a raster file."""
    file_extension = file_path.split(".")[-1]
    if file_extension in get_supported_vector_extensions():
        data = rasterio.open(file_path)
    else:
        raise UnsupportedFileExtensionException(
            f"File extension {file_extension} is not supported. Provide a raster file."
        )
    return data


def read_vector(file_path: Path) -> gpd.GeoDataFrame:
    """Read a vector file to a GeoDataFrame."""
    file_extension = file_path.split(".")[-1]
    if file_extension in get_supported_vector_extensions():
        data = gpd.read_file(file_path)
    else:
        raise UnsupportedFileExtensionException(
            f"File extension {file_extension} is not supported. Provide a vector file."
        )
    return data


def read_as_raster(file_path: Path) -> rasterio.io.DatasetReader:
    """Read spatial data as raster."""
    data = read_spatial(file_path)
    if isinstance(data, gpd.GeoDataFrame):
        # data = rasterize(data)  # TODO: rasterize
        pass
    return data


def read_tabular(file_path: Path) -> pd.DataFrame:
    """Read tabular data as a DataFrame."""
    file_extension = file_path.split(".")[-1]
    supported_tabular_extensions = ["csv"]
    if file_extension in supported_tabular_extensions:
        data = pd.read_csv(file_path)
    else:
        raise UnsupportedFileExtensionException(
            f"File extension {file_extension} is not supported. Provide a CSV file."
        )
    return data


def write_raster(file_path: Path, raster_data: np.ndarray, raster_meta: dict) -> None:
    """Write raster data to a file."""
    with rasterio.open(file_path, "w", **raster_meta) as dst:
        dst.write(raster_data)


def write_vector(file_path: Path, data: gpd.GeoDataFrame, driver: Optional[str] = None) -> None:
    """
    Write vector data from a geodataframe.

    Args:
        file_path: File path for vector data to be written.
        data: A geodataframe containing vector data.
        driver: Driver that specifies the file format. Defaults to None, in which case
            format is tried to infer from the given file extension. If no file extension
            is in the file path, defaults to ESRI shapefile.

    Raises:
        InvalidParametervalueException: Driver is not supported or recognized.
    """
    if driver not in fiona.supported_drivers:
        raise InvalidParameterValueException(f"Driver {driver} is not supported or recognized.")
    data.to_file(file_path, driver=driver)


def write_tabular(file_path: Path, data: pd.DataFrame) -> None:
    """Write tabular data (DataFrame) to a CSV file."""
    data.to_csv(file_path, index=False)
