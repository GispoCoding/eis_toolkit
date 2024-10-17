import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import NumericValueSignException
from eis_toolkit.vector_processing.proximity_computation import proximity_computation
from tests.raster_processing.clip_test import polygon_path as polygon_path
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

sys.path.append("../..")
gdf = gpd.read_file(polygon_path)

with rasterio.open(SMALL_RASTER_PATH) as test_raster:
    raster_profile = test_raster.profile

EXPECTED_SMALL_RASTER_SHAPE = raster_profile["height"], raster_profile["width"]


@pytest.mark.parametrize(
    "geodataframe,raster_profile,expected_shape,maximum_distance,scale,scaling_range",
    [
        pytest.param(
            gdf,
            raster_profile,
            EXPECTED_SMALL_RASTER_SHAPE,
            25,
            "linear",
            (1, 0),
            id="Inversion_and_scaling_between_1_and_0",
        ),
        pytest.param(
            gdf,
            raster_profile,
            EXPECTED_SMALL_RASTER_SHAPE,
            25,
            "linear",
            (2, 1),
            id="Inversion_and_scaling_between_2_and_1",
        ),
    ],
)
def test_proximity_computation_inversion_with_expected_result(
    geodataframe, raster_profile, expected_shape, maximum_distance, scale, scaling_range
):
    """Tests if the enteries in the output matrix are between the minimum and maximum value."""

    result = proximity_computation(geodataframe, raster_profile, maximum_distance, scale, scaling_range)

    assert result.shape == expected_shape
    # Assert that all values in result within scaling_range
    assert np.all((result >= scaling_range[1]) & (result <= scaling_range[0])), "Scaling out of scaling_range"


@pytest.mark.parametrize(
    "geodataframe,raster_profile,expected_shape,maximum_distance,scale,scaling_range",
    [
        pytest.param(
            gdf,
            raster_profile,
            EXPECTED_SMALL_RASTER_SHAPE,
            25,
            "linear",
            (0, 1),
            id="Scaling_between_0_and_1",
        ),
        pytest.param(
            gdf,
            raster_profile,
            EXPECTED_SMALL_RASTER_SHAPE,
            25,
            "linear",
            (1, 2),
            id="Scaling_between_1_and_2",
        ),
    ],
)
def test_proximity_computation_with_expected_result(
    geodataframe, raster_profile, expected_shape, maximum_distance, scale, scaling_range
):
    """Tests if the enteries in the output matrix are between the minimum and maximum value."""

    result = proximity_computation(geodataframe, raster_profile, maximum_distance, scale, scaling_range)

    assert result.shape == expected_shape
    # Assert that all values in result within scaling_range
    assert np.all((result <= scaling_range[1]) & (result >= scaling_range[0])), "Scaling out of scaling_range"


def test_proximity_computation_with_expected_error():
    """Tests if an exception is raised for a negative maximum distance."""

    with pytest.raises(NumericValueSignException, match="Expected max distance to be a positive number."):
        result = proximity_computation(gdf, raster_profile, -25, "linear", (1, 0))
        assert np.all((result >= 0) & (result <= 1)), "Scaling out of scaling_range"
