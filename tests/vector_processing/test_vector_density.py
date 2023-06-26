import geopandas as gpd
import numpy as np
import pytest

from eis_toolkit.vector_processing.vector_density import vector_density
from tests.vector_processing.test_rasterize_vector import SAMPLE_OVERLAPPING_POINT_GEODATAFRAME


@pytest.mark.parametrize(
    "geodataframe,resolution,base_raster_profile,buffer_value,statistic,expected_result",
    [
        pytest.param(
            SAMPLE_OVERLAPPING_POINT_GEODATAFRAME,
            0.5,
            None,
            0.5,
            "count",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            id="Overlapping_points_with_small_buffer_count",
        ),
        pytest.param(
            SAMPLE_OVERLAPPING_POINT_GEODATAFRAME,
            0.5,
            None,
            0.5,
            "density",
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            id="Overlapping_points_with_small_buffer_density",
        ),
    ],
)
def test_vector_density_with_known_result(
    geodataframe: gpd.GeoDataFrame,
    resolution,
    base_raster_profile,
    buffer_value,
    statistic,
    expected_result,
):
    """Test vector_density."""
    out_raster_array, out_metadata = vector_density(
        geodataframe=geodataframe,
        resolution=resolution,
        base_raster_profile=base_raster_profile,
        buffer_value=buffer_value,
        statistic=statistic,
    )

    assert isinstance(out_raster_array, np.ndarray)
    assert isinstance(out_metadata, dict)

    if base_raster_profile is not None:
        for key in ("transform", "width", "height"):
            assert out_metadata[key] == base_raster_profile[key]

    np.testing.assert_array_equal(out_raster_array, expected_result)
