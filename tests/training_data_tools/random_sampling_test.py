from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import EmptyDataFrameException, NumericValueSignException
from eis_toolkit.training_data_tools.points_to_raster import points_to_raster
from eis_toolkit.training_data_tools.random_sampling import generate_negatives
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
PATH_LABELS_GPKG = test_dir.joinpath("data/remote/interpolating/interpolation_test_data_small.gpkg")

gdf = gpd.read_file(PATH_LABELS_GPKG)


@pytest.mark.parametrize("geodataframe, sample_number, random_seed", [(gdf, 10, 30)])
def test_points_to_raster(geodataframe, sample_number, random_seed):
    """Test that generate_negatives function works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
        raster_profile = temp_raster.profile

        outarray, outmeta = points_to_raster(
            geodataframe=geodataframe, attribute="value", raster_profile=raster_profile
        )

        sampled_negatives, outarray2, outmeta2 = generate_negatives(
            raster_array=outarray, raster_profile=outmeta, sample_number=sample_number, random_seed=random_seed
        )

        row, col = rasterio.transform.rowcol(
            outmeta2["transform"], sampled_negatives.geometry.x, sampled_negatives.geometry.y
        )

        assert outarray2.shape == (
            temp_raster.height,
            temp_raster.width,
        ), f"Expected output array shape {(temp_raster.height, temp_raster.width)} but got {outarray2.shape}"

        assert (outarray2[row, col] == -1).all()


@pytest.mark.parametrize("geodataframe, sample_number, random_seed", [(gdf, 10, 30)])
def test_Empty_Data_Frame_exception(geodataframe, sample_number, random_seed):
    """Test that generate_negatives function raises EmptyDataFrameException for an empty raster array."""
    with pytest.raises(EmptyDataFrameException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
            raster_profile = temp_raster.profile

            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="value", raster_profile=raster_profile
            )

            outarray = np.array([])

            sampled_negatives, outarray2, outmeta2 = generate_negatives(
                raster_array=outarray, raster_profile=outmeta, sample_number=sample_number, random_seed=random_seed
            )


@pytest.mark.parametrize("geodataframe, sample_number, random_seed", [(gdf, -10, 30), (gdf, 0, 30)])
def test_Numeric_value_sign_exception(geodataframe, sample_number, random_seed):
    """Test that generate_negatives function raises NumericValueSignException for negative and zero sample number."""
    with pytest.raises(NumericValueSignException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
            raster_profile = temp_raster.profile

            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="value", raster_profile=raster_profile
            )

            sampled_negatives, outarray2, outmeta2 = generate_negatives(
                raster_array=outarray, raster_profile=outmeta, sample_number=sample_number, random_seed=random_seed
            )
