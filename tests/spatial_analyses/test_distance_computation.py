import geopandas as gpd

# import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio

# from rasterio.plot import show
from shapely.geometry import Point

from eis_toolkit.spatial_analyses.distance_computation import distance_computation
from tests.clip_test import raster_path as SMALL_RASTER_PATH

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
    "raster_path,geometries",
    [(SMALL_RASTER_PATH, GEOMETRIES_WITHIN_SMALL_RASTER)],
)
def test_distance_computation(raster_path, geometries):
    """Test distance_computation."""

    with rasterio.open(raster_path) as raster:
        result = distance_computation(raster_profile=raster.profile, geometries=geometries)

    assert isinstance(result, np.ndarray)

    # fig = plt.figure(figsize=(10, 10))
    # subfigs = fig.subfigures(2)
    # fig_1 = subfigs[0]
    # ax_1 = fig_1.add_subplot()
    # rasterio_ax = show(result, transform=SMALL_RASTER_PROFILE["transform"], ax=ax_1)
    # fig.colorbar(rasterio_ax.get_images()[0])
    # geometries.plot(ax=ax_1, alpha=1.0, color="black")

    # fig_2 = subfigs[0]
    # ax_2 = fig_2.add_subplot()
    # with rasterio.open(SMALL_RASTER_PATH) as raster:
    #     rasterio_ax = show(raster.read(), transform=SMALL_RASTER_PROFILE["transform"], ax=ax_2)

    # fig.suptitle("distance comp")
    # fig.savefig(f"/mnt/c/tmp/plot.png")
