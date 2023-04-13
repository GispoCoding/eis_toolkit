from contextlib import nullcontext

import geopandas as gpd

# import matplotlib.pyplot as plt
import numpy as np
import pytest
from rasterio import profiles, transform
from shapely.geometry import LineString, Point, box

from eis_toolkit import exceptions
from eis_toolkit.vector_processing.rasterize import rasterize_vector

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
            Point(10, 10),
        ],
        "values": [1, 2, 3],
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


@pytest.mark.parametrize(
    "geodataframe,resolution,value_column,default_value,fill_value,base_raster_profile,buffer_value,raises",
    [
        pytest.param(
            SAMPLE_LINE_GEODATAFRAME,
            0.05,
            None,
            1,
            0,
            None,
            None,
            nullcontext(),
            id="LineStrings",
        ),
        pytest.param(
            SAMPLE_POINT_GEODATAFRAME,
            0.5,
            None,
            1,
            0,
            None,
            None,
            nullcontext(),
            id="Points",
        ),
        pytest.param(
            SAMPLE_POLYGON_GEODATAFRAME,
            0.5,
            None,
            1,
            0,
            None,
            None,
            nullcontext(),
            id="Polygons",
        ),
        pytest.param(
            SAMPLE_EMPTY_GEODATAFRAME,
            0.5,
            None,
            1,
            0,
            None,
            None,
            pytest.raises(exceptions.EmptyDataFrameException),
            id="Empty_GeoDataFrame_that_should_raise_exception",
        ),
        pytest.param(
            SAMPLE_TRACES_WITH_EMPTY_GEODATAFRAME,
            0.5,
            None,
            1,
            0,
            None,
            None,
            nullcontext(),
            id="LineStrings_with_some_empty",
        ),
        pytest.param(
            SAMPLE_LINE_GEODATAFRAME,
            0.15,
            None,
            1,
            0,
            profiles.Profile(
                dict(height=20, width=20, transform=transform.from_bounds(-10, -10, 10, 10, width=20, height=20))
            ),
            None,
            nullcontext(),
            id="LineStrings_with_base_raster",
        ),
        pytest.param(
            SAMPLE_LINE_GEODATAFRAME,
            0.15,
            None,
            1,
            0,
            None,
            1.0,
            nullcontext(),
            id="LineStrings_with_buffer",
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

    # # with rasterio.open(
    # #         "/mnt/c/tmp/new.tiff",
    # #         mode="w",
    # #         driver="GTiff",
    # #         height=out_raster_array.shape[0],
    # #         width=out_raster_array.shape[1],
    # #         count=1,
    # #         dtype=out_raster_array.dtype,
    # #         crs=geodataframe.crs,
    # #         transform=out_metadata["transform"]
    # #         ) as new_dataset:

    # #     new_dataset.write(out_raster_array, 1)
    # # geodataframe.to_file("/mnt/c/tmp/new.gpkg", driver="GPKG")

    # fig, ax = plt.subplots()
    # rasterio_ax = plot.show(out_raster_array, transform=out_metadata["transform"], ax=ax)
    # fig.colorbar(rasterio_ax.get_images()[0])
    # geodataframe.plot(ax=ax, alpha=0.5)

    # # min_x, min_y, max_x, max_y = geodataframe.total_bounds
    # # ax.set_xlim(min_x - (0.1 * max_x), max_x * 1.1)
    # # ax.set_ylim(min_y - (0.1 * max_y), max_y * 1.1)
    # name = f"{geodataframe.geometry.values[0].geom_type}
    # _{resolution}_{value_column}_{base_raster_profile is None}_{buffer_value is not None}"
    # fig.suptitle(name)
    # fig.savefig(f"/mnt/c/tmp/new_{name}.png")
