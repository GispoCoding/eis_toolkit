from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.conversions.raster_to_dataframe import raster_to_dataframe
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent


# @pytest.mark.skip
def test_raster_to_dataframe():
    """Test raster to pandas conversion by converting pandas dataframe and then back to raster data."""
    raster = rasterio.open(SMALL_RASTER_PATH)
    raster_data_array = raster.read(1)

    """Create multiband raster for testing purposes."""
    multiband_path = test_dir.joinpath("data/local/data/multiband.tif")
    meta = raster.meta.copy()
    meta["count"] = 4
    with rasterio.open(multiband_path, "w", **meta) as dest:
        for band in range(1, 5):
            dest.write(raster_data_array - band, band)

    """Convert to dataframe."""
    multiband_raster = rasterio.open(test_dir.joinpath("data/local/data/multiband.tif"))
    df = raster_to_dataframe(multiband_raster, add_coordinates=True)

    """Convert back to raster image."""
    df["id"] = df.index
    long_df = pd.wide_to_long(df, ["band_"], i="id", j="band").reset_index()
    long_df.loc[:, ["col", "row"]] = long_df.loc[:, ["col", "row"]].astype(int)
    raster_img = np.empty((multiband_raster.count, multiband_raster.height, multiband_raster.width))
    raster_img[(long_df.band - 1).to_list(), long_df.row.to_list(), long_df.col.to_list()] = long_df.band_

    assert np.array_equal(multiband_raster.read(), raster_img)
