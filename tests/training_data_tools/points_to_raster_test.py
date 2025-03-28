from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidColumnException,
    NonMatchingCrsException,
    NonNumericDataException,
)
from eis_toolkit.training_data_tools.points_to_raster import points_to_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
PATH_LABELS_GPKG = test_dir.joinpath("data/remote/interpolating/interpolation_test_data_small.gpkg")

geodataframe = gpd.read_file(PATH_LABELS_GPKG)
gdf = gpd.GeoDataFrame()


@pytest.mark.parametrize("geodataframe", [geodataframe])  # Case where CRS matches
def test_points_to_raster(geodataframe):
    """Test that points_to_raster function works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

        raster_profile = temp_raster.profile

        outarray, outmeta = points_to_raster(
            geodataframe=geodataframe, attribute="value", raster_profile=raster_profile
        )

        assert outarray.shape == (
            temp_raster.height,
            temp_raster.width,
        ), f"Expected output array shape {(temp_raster.height,temp_raster.width)} but got {outarray.shape}"


@pytest.mark.parametrize("geodataframe", [geodataframe])
def test_InvalidColumnException(geodataframe):
    """Test that incorrect attribute raises InvalidColumnException."""
    with pytest.raises(InvalidColumnException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

            raster_profile = temp_raster.profile

            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="data", raster_profile=raster_profile
            )


@pytest.mark.parametrize("geodataframe", [geodataframe])
def test_Nonnumeric_Data(geodataframe):
    """Test that non numeric values in attribute column raises NonNumericDataException."""
    with pytest.raises(NonNumericDataException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

            raster_profile = temp_raster.profile

            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="id", raster_profile=raster_profile
            )


@pytest.mark.parametrize("geodataframe", [gdf])
def test_Empty_Dataframe(geodataframe):
    """Test that empty geodataframe raises EmptyDataFrameException."""
    with pytest.raises(EmptyDataFrameException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

            raster_profile = temp_raster.profile

            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="id", raster_profile=raster_profile
            )


@pytest.mark.parametrize("geodataframe", [geodataframe.to_crs(epsg=4326)])  # Case where CRS do not matches
def test_non_matching_crs_error(geodataframe):
    """Test that different crs raises NonMatchingCrsException."""

    with pytest.raises(NonMatchingCrsException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
            raster_profile = temp_raster.profile
            outarray, outmeta = points_to_raster(
                geodataframe=geodataframe, attribute="value", raster_profile=raster_profile
            )
