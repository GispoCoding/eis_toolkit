from contextlib import nullcontext

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from eis_toolkit import exceptions
from eis_toolkit.vector_processing.distance_computation import distance_computation
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

with rasterio.open(SMALL_RASTER_PATH) as raster:
    SMALL_RASTER_PROFILE = raster.profile

GEOMETRIES_WITHIN_SMALL_RASTER = gpd.GeoDataFrame(
    {
        "geometry": [
            Point(384745.000, 6671375.000),
            Point(384800.000, 6671375.000),
        ]
    },
    crs=SMALL_RASTER_PROFILE["crs"],
)


@pytest.mark.parametrize(
    "raster_profile,geometries,expected_shape,expected_min,expected_max",
    [
        (
            SMALL_RASTER_PROFILE,
            GEOMETRIES_WITHIN_SMALL_RASTER,
            (56, 46),
            0.0,
            107.83784122468327,
        )
    ],
)
def test_distance_computation_with_expected_results(
    raster_profile, geometries, expected_shape, expected_min, expected_max
):
    """Test distance_computation."""

    result = distance_computation(raster_profile=raster_profile, geometries=geometries)

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape
    assert np.isclose(result.min(), expected_min)
    assert np.isclose(result.max(), expected_max)


@pytest.mark.parametrize(
    "raster_profile,geometries,expected_error",
    [
        (
            SMALL_RASTER_PROFILE,
            GEOMETRIES_WITHIN_SMALL_RASTER,
            nullcontext(),
        ),
        (
            {**SMALL_RASTER_PROFILE, "height": None},
            GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(exceptions.InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "height": 0.123},
            GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(exceptions.InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "width": 0.123},
            GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(exceptions.InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "transform": None},
            GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(exceptions.InvalidParameterValueException),
        ),
    ],
)
def test_distance_computation(raster_profile, geometries, expected_error):
    """Test distance_computation."""

    with expected_error:
        result = distance_computation(raster_profile=raster_profile, geometries=geometries)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
