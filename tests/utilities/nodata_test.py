from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.utilities.nodata import (
    replace_nodata_dataframe,
    replace_raster_nodata_each_band,
    replace_values_with_nodata,
    set_nodata_raster_meta,
)

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
multi_raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_set_nodata_raster_meta():
    """Test that setting raster nodata works as intended."""
    raster = rasterio.open(raster_path)
    new_meta = set_nodata_raster_meta(raster.meta, 10)
    assert new_meta["nodata"] == 10


def test_replace_values_with_nodata_1():
    """Test that replacing specified values with nodata in a Numpy array works as expected. Case 1."""
    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    target_arr = np.array([[1, np.nan, np.nan, np.nan, 1], [np.nan, np.nan, 5, 4, np.nan]])
    values_to_replace = [2, 3]
    replaced_arr = replace_values_with_nodata(data, values_to_replace)
    assert np.allclose(replaced_arr, target_arr, equal_nan=True)


def test_replace_values_with_nodata_2():
    """Test that replacing specified values with nodata in a Numpy array works as expected. Case 2."""
    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    target_arr = np.array([[np.nan, 2, 3, 2, np.nan], [2, 3, 5, 4, 3]])
    values_to_replace = 1
    replaced_arr = replace_values_with_nodata(data, values_to_replace)
    assert np.allclose(replaced_arr, target_arr, equal_nan=True)


def test_replace_raster_nodata_each_band():
    """Test that replacing replacing nodata for each band separately works as expected."""
    multiband_data = np.array([[[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]], [[15, 16, 16, 17, 12], [16, 13, 10, 14, 16]]])
    nodata_per_band = {1: 2, 2: [16, 17]}
    target_data = np.array(
        [[[1, -9999, 3, -9999, 1], [-9999, 3, 5, 4, 3]], [[15, -9999, -9999, -9999, 12], [-9999, 13, 10, 14, -9999]]]
    )
    bands_replaced = replace_raster_nodata_each_band(multiband_data, nodata_per_band)
    assert np.allclose(bands_replaced, target_data)


def test_replace_nodata_dataframe():
    """Test that replacing specified values with nodata in a dataframe works as expected."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 5, 6]})
    target_df = pd.DataFrame({"col1": [1, 2, np.nan], "col2": [np.nan, 5, 6]})
    replaced_df = replace_nodata_dataframe(df, old_nodata=3)
    assert replaced_df.equals(target_df)
