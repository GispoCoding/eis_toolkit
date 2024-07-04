import tempfile

import geopandas as gpd
import numpy as np
import rasterio

from eis_toolkit.raster_processing.clipping import clip_raster
from eis_toolkit.raster_processing.masking import mask_raster
from tests.raster_processing.clip_test import polygon_path as SMALL_VECTOR_PATH
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH


def test_mask_raster():
    """Test that masking raster works as intended."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        geodataframe = gpd.read_file(SMALL_VECTOR_PATH)
        out_image, out_meta = clip_raster(raster, geodataframe)
        with tempfile.NamedTemporaryFile() as tmpfile:
            with rasterio.open(tmpfile.name, "w", **out_meta) as dest:
                dest.write(out_image)
            with rasterio.open(tmpfile.name) as base_raster:
                old_nodata_count = np.count_nonzero(raster.read(1) == raster.nodata)
                out_image, out_profile = mask_raster(raster, base_raster)

                new_nodata_count = np.count_nonzero(out_image == out_profile["nodata"])

                # Check nodata count has increased
                assert new_nodata_count > old_nodata_count
                # Check that nodata exists now in identical locations in input raster and base raster
                np.testing.assert_array_equal(
                    base_raster.read(1) == base_raster.nodata, out_image[0] == out_profile["nodata"]
                )
