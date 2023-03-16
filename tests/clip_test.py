from pathlib import Path

import geopandas
import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException
from eis_toolkit.raster_processing.clipping import clip_raster

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
polygon_path = parent_dir.joinpath("data/remote/small_area.shp")
point_path = parent_dir.joinpath("data/remote/point.gpkg")
wrong_crs_polygon_path = parent_dir.joinpath("data/remote/small_area.geojson")

# Save output to local to not push it
output_raster_path = parent_dir.joinpath("data/local/test.tif")


def test_clip_raster():
    """Test clip functionality with geotiff raster and shapefile polygon."""
    geodataframe = geopandas.read_file(polygon_path)

    with rasterio.open(raster_path) as raster:
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    with rasterio.open(output_raster_path) as result:
        assert np.amax(result.read()) != np.amin(result.read())
        assert result.count == 1
        # Corresponding operation executed with QGIS for a reference solution
        assert result.height == result.width == 26
        assert result.bounds[0] == 384760.0
        assert result.bounds[3] == 6671364.0


def test_clip_raster_wrong_geometry_type():
    """Tests that non-polygon geometry raises the correct exception."""
    with pytest.raises(NotApplicableGeometryTypeException):
        point = geopandas.read_file(point_path)
        with rasterio.open(raster_path) as raster:
            clip_raster(
                raster=raster,
                geodataframe=point,
            )


def test_clip_raster_different_crs():
    """Test that a crs mismatch raises the correct exception."""
    with pytest.raises(NonMatchingCrsException):
        wrong_crs_polygon = geopandas.read_file(wrong_crs_polygon_path)
        with rasterio.open(raster_path) as raster:
            clip_raster(
                raster=raster,
                geodataframe=wrong_crs_polygon,
            )
