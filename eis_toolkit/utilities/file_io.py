import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def read_raster(file_path: str) -> rasterio.io.DatasetReader:
    """Read and open a raster file."""
    return rasterio.open(file_path)


def write_raster(file_path: str, raster_data: np.ndarray, raster_meta: dict) -> None:
    """Write raster data to a file."""
    with rasterio.open(file_path, "w", **raster_meta) as dst:
        dst.write(raster_data)


def read_geojson(file_path: str) -> gpd.GeoDataFrame:
    """Read a GeoJSON to a GeoDataFrame."""
    return gpd.read_file(file_path)


def write_geojson(file_path: str, data: gpd.GeoDataFrame) -> None:
    """Read vector data (GeoDataFrame) to a GeoJSON file."""
    data.to_file(file_path, driver="GeoJSON")


def read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file to a DataFrame."""
    return pd.read_csv(file_path)


def write_csv(file_path: str, data: pd.DataFrame) -> None:
    """Write tabular data (DataFrame) to a CSV file."""
    data.to_csv(file_path, index=False)
