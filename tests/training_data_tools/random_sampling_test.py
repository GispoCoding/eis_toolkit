from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

from eis_toolkit.training_data_tools.points_to_raster import points_to_raster
from eis_toolkit.training_data_tools.random_sampling import generate_negatives
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
PATH_LABELS_GPKG = test_dir.joinpath("data/remote/interpolating/interpolation_test_data_small.gpkg")

gdf = gpd.read_file(PATH_LABELS_GPKG)


@pytest.mark.parametrize("geodataframe", [gdf])
def test_points_to_raster(geodataframe):
    """Test that generate_negatives function works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
        raster_profile = temp_raster.profile

        outarray, outmeta = points_to_raster(geodataframe=gdf, attribute="value", raster_profile=raster_profile)

        sampled_negatives, outarray2, outmeta2 = generate_negatives(
            raster_array=outarray, raster_meta=outmeta, sample_number=10, random_seed=30
        )

        row, col = rasterio.transform.rowcol(
            outmeta2["transform"], sampled_negatives.geometry.x, sampled_negatives.geometry.y
        )

        assert outarray2.shape == (
            temp_raster.height,
            temp_raster.width,
        ), f"Expected output array shape {(temp_raster.height, temp_raster.width)} but got {outarray2.shape}"

        assert (outarray2[row, col] == -1).all()
