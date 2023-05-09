from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from pandas.testing import assert_series_equal

from eis_toolkit.raster_processing.extract_values_from_raster import extract_values_from_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
gdf_path = test_dir.joinpath("data/remote/extract_raster_values/extract_raster_values_points.shp")


def test_extract_values_from_raster_returns_correct_output():
    """Test extract raster values returns correct output."""
    expected_output = pd.Series([5.683, 5.040, 2.958, 8.799, 5.234], name="small_raster")

    single_band_raster = rasterio.open(SMALL_RASTER_PATH)
    gdf = gpd.read_file(gdf_path)

    raster_list = [single_band_raster]

    data_frame = extract_values_from_raster(raster_list=raster_list, geodataframe=gdf)
    data_frame_column = data_frame["small_raster"].squeeze()

    assert_series_equal(data_frame_column, expected_output)


def test_extract_values_from_raster_returns_pandas_dataframe():
    """Test extract raster values returns pandas DataFrame."""
    single_band_raster = rasterio.open(SMALL_RASTER_PATH)
    gdf = gpd.read_file(gdf_path)

    raster_list = [single_band_raster]

    data_frame = extract_values_from_raster(raster_list=raster_list, geodataframe=gdf)

    assert isinstance(data_frame, pd.DataFrame)


def test_extract_values_from_raster_returns_non_empty_dataframe():
    """Test extract raster values returns a filled pandas DataFrame."""
    single_band_raster = rasterio.open(SMALL_RASTER_PATH)
    gdf = gpd.read_file(gdf_path)

    raster_list = [single_band_raster]

    data_frame = extract_values_from_raster(raster_list=raster_list, geodataframe=gdf)

    assert not data_frame.empty


def test_extract_values_from_raster_uses_custom_column_names():
    """Test extract raster values full pandas DataFrame."""
    single_band_raster = rasterio.open(SMALL_RASTER_PATH)
    gdf = gpd.read_file(gdf_path)

    raster_list = [single_band_raster]

    column_name = "singleband_raster"
    raster_column_names = [column_name]

    data_frame = extract_values_from_raster(
        raster_list=raster_list, geodataframe=gdf, raster_column_names=raster_column_names
    )

    assert column_name in data_frame.columns
