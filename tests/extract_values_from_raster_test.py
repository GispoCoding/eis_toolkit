from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import rasterio

from eis_toolkit.raster_processing.extract_values_from_raster import extract_values_from_raster
from eis_toolkit.exceptions import InvalidParameterValueException

parent_dir = Path(__file__).parent
singleband_raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
shapefile_path = parent_dir.joinpath("data/remote/extract_raster_values/extract_raster_values_point.shp")

def test_extract_values_from_raster_returns_pandas_dataframe():
    '''Test extract raster values returns pandas DataFrame'''
    single_band_raster = rasterio.open(singleband_raster_path)
    shapefile = gpd.read_file(shapefile_path)

    raster_list = [single_band_raster]

    data_frame = extract_values_from_raster(raster_list = raster_list, shapefile = shapefile)

    assert isinstance(data_frame, pd.DataFrame)

def test_extract_values_from_raster_returns_non_empty_dataframe():
    '''Test extract raster values returns a filled pandas DataFrame'''
    single_band_raster = rasterio.open(singleband_raster_path)
    shapefile = gpd.read_file(shapefile_path)

    raster_list = [single_band_raster]

    data_frame = extract_values_from_raster(raster_list = raster_list, shapefile = shapefile)

    assert not data_frame.empty

def test_extract_values_from_raster_uses_custom_column_names():
    '''Test extract raster values full pandas DataFrame'''
    single_band_raster = rasterio.open(singleband_raster_path)
    shapefile = gpd.read_file(shapefile_path)

    raster_list = [single_band_raster]

    column_name = "singleband_raster"
    raster_column_names = [column_name]

    data_frame = extract_values_from_raster(raster_list = raster_list, shapefile = shapefile, raster_column_names = raster_column_names)

    assert column_name in data_frame.columns

def test_extract_values_from_raster_empty_column_names_raises_invalid_parameter_exception():
    """Test that invalid parameter raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(singleband_raster_path) as single_band_raster:
            raster_list = [single_band_raster]
            shapefile = gpd.read_file(shapefile_path)
            raster_column_names = []
            extract_values_from_raster(raster_list, shapefile, raster_column_names)

def test_extract_values_from_raster_numeric_column_names_raises_invalid_parameter_exception():
    """Test that invalid parameter raises the correct exception."""    
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(singleband_raster_path) as single_band_raster:
            raster_list = [single_band_raster]
            shapefile = gpd.read_file(shapefile_path)
            raster_column_names = [1]
            extract_values_from_raster(raster_list, shapefile, raster_column_names)
