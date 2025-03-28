from contextlib import nullcontext
from functools import partial

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, Point, box

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.vector_processing.distance_computation import distance_computation
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

with rasterio.open(SMALL_RASTER_PATH) as raster:
    SMALL_RASTER_PROFILE = raster.profile

geodataframe_with_raster_crs = partial(gpd.GeoDataFrame, crs=SMALL_RASTER_PROFILE["crs"])

EXPECTED_SMALL_RASTER_SHAPE = SMALL_RASTER_PROFILE["height"], SMALL_RASTER_PROFILE["width"]


POINT_GEOMETRIES_WITHIN_SMALL_RASTER = geodataframe_with_raster_crs(
    geometry=[
        Point(384745.000, 6671375.000),
        Point(384800.000, 6671375.000),
    ]
)
LINE_GEOMETRIES_WITHIN_SMALL_RASTER = geodataframe_with_raster_crs(
    geometry=[
        LineString([Point(384745.000, 6671375.000), Point(384800.000, 6671375.000)]),
        LineString([Point(384745.000, 6671375.000), Point(384745.000, 6671375.000)]),
    ]
)
POLYGON_GEOMETRIES_WITHIN_SMALL_RASTER = geodataframe_with_raster_crs(
    geometry=[
        box(384744.000, 6671272.000, 384764.000, 6671292.000),
        box(384784.000, 6671280.000, 384800.000, 6671300.000),
    ]
)


@pytest.mark.parametrize(
    "raster_profile,geodataframe,expected_shape,expected_min,expected_max",
    [
        pytest.param(
            SMALL_RASTER_PROFILE,
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            EXPECTED_SMALL_RASTER_SHAPE,
            0.0,
            107.51744,
            id="point_geometries_within_small_raster",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            LINE_GEOMETRIES_WITHIN_SMALL_RASTER,
            EXPECTED_SMALL_RASTER_SHAPE,
            0.0,
            107.51744,
            id="line_geometries_within_small_raster",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            POLYGON_GEOMETRIES_WITHIN_SMALL_RASTER,
            EXPECTED_SMALL_RASTER_SHAPE,
            0.0,
            92.0,
            id="polygon_geometries_within_small_raster",
        ),
    ],
)
def test_distance_computation_with_expected_results(
    raster_profile, geodataframe, expected_shape, expected_min, expected_max
):
    """Test distance_computation."""

    out_image, _ = distance_computation(raster_profile=raster_profile, geodataframe=geodataframe)

    assert isinstance(out_image, np.ndarray)
    assert out_image.shape == expected_shape
    assert np.isclose(out_image.min(), expected_min, atol=0.0001)
    assert np.isclose(out_image.max(), expected_max, atol=0.0001)


@pytest.mark.parametrize(
    "raster_profile,geodataframe,expected_error",
    [
        (
            SMALL_RASTER_PROFILE,
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            nullcontext(),
        ),
        (
            {**SMALL_RASTER_PROFILE, "height": None},
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "height": 0.123},
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "width": 0.123},
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(InvalidParameterValueException),
        ),
        (
            {**SMALL_RASTER_PROFILE, "transform": None},
            POINT_GEOMETRIES_WITHIN_SMALL_RASTER,
            pytest.raises(InvalidParameterValueException),
        ),
    ],
)
def test_distance_computation(raster_profile, geodataframe, expected_error):
    """Test distance_computation."""

    with expected_error:
        out_image, _ = distance_computation(raster_profile=raster_profile, geodataframe=geodataframe)
        assert isinstance(out_image, np.ndarray)
        assert len(out_image.shape) == 2
