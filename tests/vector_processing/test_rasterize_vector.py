from contextlib import nullcontext
from typing import NamedTuple, Optional, Union

import geopandas as gpd
import numpy as np
import pytest
from rasterio import profiles, transform
from shapely.geometry import LineString, Point, box

from eis_toolkit import exceptions
from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

SAMPLE_LINE_GEODATAFRAME = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 1), (1, 2)]),
            LineString([(0, 2), (1, 3)]),
        ],
        "values": [1, 2, 3],
    },
)

SAMPLE_POINT_GEODATAFRAME = gpd.GeoDataFrame(
    {
        "geometry": [
            Point(0, 0),
            Point(2, 2),
            Point(3, 3),
            Point(4, 2),
            Point(5, 5),
        ],
        "values": [1, 2, 3, 4, 5],
    },
)

SAMPLE_POLYGON_GEODATAFRAME = gpd.GeoDataFrame(
    {
        "geometry": [
            box(0, 0, 2, 2),
            Point(5, 5).buffer(2.0),
            LineString([(6, 6), (10, 10)]).buffer(0.5),
        ],
        "values": [1, 2, 3],
    },
)

SAMPLE_EMPTY_GEODATAFRAME = gpd.GeoDataFrame(
    {
        "geometry": [],
        "values": [],
    },
)

SAMPLE_TRACES_WITH_EMPTY_GEODATAFRAME = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString(),
            LineString([(0, 1), (1, 2)]),
            LineString([(0, 2), (1, 3)]),
        ],
        "values": [1, 2, 3],
    },
)


class RasterizeVectorTestArgs(NamedTuple):
    """Tuple for holding arguments, with defaults, for testing rasterize_vector."""

    geodataframe: gpd.GeoDataFrame
    resolution: float = 1.0
    value_column: Optional[str] = None
    default_value: float = 1.0
    fill_value: float = 0.0
    base_raster_profile: Optional[Union[profiles.Profile, dict]] = None
    buffer_value: Optional[float] = None


@pytest.mark.parametrize(
    "geodataframe,resolution,value_column,default_value,fill_value,base_raster_profile,buffer_value,raises",
    [
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_LINE_GEODATAFRAME, resolution=0.05),
            nullcontext(),
            id="LineStrings",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_POINT_GEODATAFRAME, resolution=0.5),
            nullcontext(),
            id="Points",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(
                geodataframe=SAMPLE_POINT_GEODATAFRAME,
                resolution=-0.5,
            ),
            pytest.raises(exceptions.NumericValueSignException),
            id="Points_with_negative_resolution",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_POLYGON_GEODATAFRAME),
            nullcontext(),
            id="Polygons",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_EMPTY_GEODATAFRAME),
            pytest.raises(exceptions.EmptyDataFrameException),
            id="Empty_GeoDataFrame_that_should_raise_exception",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_TRACES_WITH_EMPTY_GEODATAFRAME),
            nullcontext(),
            id="LineStrings_with_some_empty",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(
                geodataframe=SAMPLE_LINE_GEODATAFRAME,
                base_raster_profile=profiles.Profile(
                    dict(height=20, width=20, transform=transform.from_bounds(-10, -10, 10, 10, width=20, height=20))
                ),
            ),
            nullcontext(),
            id="LineStrings_with_base_raster",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(geodataframe=SAMPLE_LINE_GEODATAFRAME, buffer_value=1.0, resolution=0.15),
            nullcontext(),
            id="LineStrings_with_buffer",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(
                geodataframe=SAMPLE_LINE_GEODATAFRAME,
                value_column="not-in-columns",
            ),
            pytest.raises(exceptions.InvalidParameterValueException),
            id="Invalid_value_column",
        ),
    ],
)
def test_rasterize_vector(
    geodataframe: gpd.GeoDataFrame,
    resolution,
    value_column,
    default_value,
    fill_value,
    base_raster_profile,
    raises,
    buffer_value,
):
    """Test rasterize_vector."""
    results = None
    with raises:
        results = rasterize_vector(
            geodataframe=geodataframe,
            resolution=resolution,
            value_column=value_column,
            default_value=default_value,
            fill_value=fill_value,
            base_raster_profile=base_raster_profile,
            buffer_value=buffer_value,
        )

    if results is None:
        # Expected exception was raised
        return
    out_raster_array, out_metadata = results
    assert isinstance(out_raster_array, np.ndarray)
    assert isinstance(out_metadata, dict)

    if base_raster_profile is not None:
        for key in ("transform", "width", "height"):
            assert out_metadata[key] == base_raster_profile[key]


@pytest.mark.parametrize(
    "geodataframe,resolution,value_column,default_value,fill_value,base_raster_profile,buffer_value,expected_result",
    [
        pytest.param(
            *RasterizeVectorTestArgs(
                geodataframe=SAMPLE_LINE_GEODATAFRAME,
                resolution=0.25,
                value_column=None,
                default_value=1.0,
                fill_value=0.0,
                base_raster_profile=None,
                buffer_value=None,
            ),
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                ],
                dtype=np.uint8,
            ),
            id="LineStrings",
        ),
        pytest.param(
            *RasterizeVectorTestArgs(
                geodataframe=SAMPLE_POINT_GEODATAFRAME,
                resolution=0.5,
                value_column=None,
                default_value=1.0,
                fill_value=0.0,
                base_raster_profile=None,
                buffer_value=0.5,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            id="Points_with_small_buffer",
        ),
    ],
)
def test_rasterize_vector_with_known_result(
    geodataframe: gpd.GeoDataFrame,
    resolution,
    value_column,
    default_value,
    fill_value,
    base_raster_profile,
    buffer_value,
    expected_result,
):
    """Test rasterize_vector."""
    out_raster_array, out_metadata = rasterize_vector(
        geodataframe=geodataframe,
        resolution=resolution,
        value_column=value_column,
        default_value=default_value,
        fill_value=fill_value,
        base_raster_profile=base_raster_profile,
        buffer_value=buffer_value,
    )

    assert isinstance(out_raster_array, np.ndarray)
    assert isinstance(out_metadata, dict)

    if base_raster_profile is not None:
        for key in ("transform", "width", "height"):
            assert out_metadata[key] == base_raster_profile[key]

    np.testing.assert_array_equal(out_raster_array, expected_result)
