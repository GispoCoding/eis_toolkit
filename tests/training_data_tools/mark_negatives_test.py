from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException
from eis_toolkit.training_data_tools.mark_negatives import mark_negatives
from eis_toolkit.training_data_tools.points_to_raster import points_to_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
PATH_LABELS_GPKG = test_dir.joinpath("data/remote/interpolating/interpolation_test_data_small.gpkg")
PATH_NEGATIVES_GPKG = test_dir.joinpath("data/remote/point.gpkg")

positives = gpd.read_file(PATH_LABELS_GPKG)
negatives = gpd.read_file(PATH_NEGATIVES_GPKG)


@pytest.mark.parametrize("positives, negatives", [(positives, negatives)])  # Pass both GeoDataFrames as a tuple
def test_points_to_raster(positives, negatives):
    """Test that mark_negatives function works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as temp_raster:

        outarray, outmeta = points_to_raster(
            positives=positives, attribute="value", template_raster=temp_raster, nodata_value=-999
        )

        outarray, outmeta = mark_negatives(negatives=negatives, raster_array=outarray, raster_meta=outmeta)

        assert outarray.shape == (
            temp_raster.height,
            temp_raster.width,
        ), f"Expected output array shape {(temp_raster.height, temp_raster.width)} but got {outarray.shape}"


@pytest.mark.parametrize(
    "positives, negatives", [(positives, negatives.to_crs(epsg=4326))]
)  # Case where CRS do not matches
def test_non_matching_crs_error(positives, negatives):
    """Test that different crs raises NonMatchingCrsException."""

    with pytest.raises(NonMatchingCrsException):
        with rasterio.open(SMALL_RASTER_PATH) as temp_raster:
            outarray, outmeta = points_to_raster(
                positives=positives, attribute="value", template_raster=temp_raster, nodata_value=-999
            )
            outarray, outmeta = mark_negatives(negatives=negatives, raster_array=outarray, raster_meta=outmeta)
