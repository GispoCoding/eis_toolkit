from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.conversions.raster_to_pandas import raster_to_pandas
from eis_toolkit.exceptions import InvalidParameterValueException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")


def test_raster_to_pandas():
    """Test raster to pandas conversion by converting pandas dataframe and then back to raster data."""
    raster = rasterio.open(raster_path)
    raster_data_array = raster.read(1)

    """Create multiband raster for testing purposes."""
    multiband_path = parent_dir.joinpath("data/local/data/multiband.tif")
    meta = raster.meta.copy()
    meta["count"] = 4
    with rasterio.open(multiband_path, "w", **meta) as dest:
        for band in range(1, 5):
            dest.write(raster_data_array - band, band)

    """Convert to dataframe."""
    multiband_raster = rasterio.open(parent_dir.joinpath("data/local/data/multiband.tif"))
    df = raster_to_pandas(multiband_raster, add_img_coord=True)

    """Convert back to raster image."""
    df["id"] = df.index
    long_df = pd.wide_to_long(df, ["band_"], i="id", j="band").reset_index()
    long_df.loc[:, ["col", "row"]] = long_df.loc[:, ["col", "row"]].astype(int)
    raster_img = np.empty((multiband_raster.count, multiband_raster.height, multiband_raster.width))
    raster_img[long_df.band - 1, long_df.row, long_df.col] = long_df.band_

    assert np.array_equal(multiband_raster.read(), raster_img)


def test_raster_to_pandas_invalid_parameter_value():
    """Test that invalid parameter value for bands raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            raster_to_pandas(raster, bands=["1", "2"])
