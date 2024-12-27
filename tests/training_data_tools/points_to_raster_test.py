import geopandas as gpd
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException
from eis_toolkit.training_data_tools.points_to_raster import points_to_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

# Generating data
positives = pd.DataFrame(
    {
        "x": [384824, 384803, 384807, 384793, 384773, 384785],
        "y": [6671284, 6671295, 6671277, 6671293, 6671343, 6671357],
    }
)
positives = gpd.GeoDataFrame(positives, geometry=gpd.points_from_xy(positives.x, positives.y, crs="EPSG:3067"))


@pytest.mark.parametrize("positives", [positives])  # Case where CRS matches
def test_points_to_raster(positives):
    """Test that points_to_raster function works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

        outarray, outmeta = points_to_raster(positives, temp_raster, nodata_value=-999)

        assert outarray.shape == (
            temp_raster.height,
            temp_raster.width,
        ), f"Expected output array shape {(temp_raster.height,temp_raster.width)} but got {outarray.shape}"


@pytest.mark.parametrize("positives", [positives.to_crs(epsg=4326)])  # Case where CRS do not matches
def test_non_matching_crs_error(positives):
    """Test that different crs raises NonMatchingCrsException."""

    with pytest.raises(NonMatchingCrsException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
            outarray, outmeta = points_to_raster(positives=positives, template_raster=temp_raster, nodata_value=-999)
