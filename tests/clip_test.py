import pytest
import numpy as np
from pathlib import Path
import rasterio
import geopandas
from eis_toolkit.raster_processing.clipping import clip
from eis_toolkit.exceptions import NonMatchingCrsException
from eis_toolkit.exceptions import NotApplicableGeometryTypeException


parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
polygon_path = parent_dir.joinpath("data/remote/small_area.shp")
point_path = parent_dir.joinpath("data/remote/point.gpkg")
wrong_crs_polygon_path = parent_dir.joinpath("data/remote/small_area.geojson")

# Save output to local to not push it
output_raster_path = parent_dir.joinpath("data/local/test.tif")


def test_clip():
    """Tests clip functionality with geotiff raster and shapefile polygon."""

    polygon = geopandas.read_file(polygon_path)
    with rasterio.open(raster_path) as raster:
        out_image, out_meta = clip(
            raster=raster,
            polygon=polygon
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


def test_clip_wrong_geometry_type():
    """Tests that non-polygon geometry raises the correct exception."""

    with pytest.raises(NotApplicableGeometryTypeException):
        point = geopandas.read_file(point_path)
        with rasterio.open(raster_path) as raster:
            clip(
                raster=raster,
                polygon=point,
            )


def test_clip_different_crs():
    """Tests that a crs mismatch raises the correct exception."""

    with pytest.raises(NonMatchingCrsException):
        wrong_crs_polygon = geopandas.read_file(wrong_crs_polygon_path)
        with rasterio.open(raster_path) as raster:
            clip(
                raster=raster,
                polygon=wrong_crs_polygon,
            )
