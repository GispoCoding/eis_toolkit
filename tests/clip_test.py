import pytest
import numpy as np
from pathlib import Path
import rasterio
import geopandas
from eis_toolkit.raster_processing.clipping import clip
from eis_toolkit.exceptions import NonMatchingCrsException
from eis_toolkit.exceptions import NotApplicableGeometryTypeException


parent_dir = Path(__file__).parent
input_raster_path = parent_dir.joinpath("data/small_raster.tif")
input_polygon_path = parent_dir.joinpath("data/small_area.shp")
output_raster_path = parent_dir.joinpath("data/test.tif")
input_point_path = parent_dir.joinpath("data/point.gpkg")
wrong_crs_input_polygon_path = parent_dir.joinpath("data/small_area.geojson")


def test_clip():
    """Tests clip functionality with geotiff raster and shapefile polygon."""

    input_polygon = geopandas.read_file(input_polygon_path)
    with rasterio.open(input_raster_path) as input_raster:
        clipped, out_transform, out_meta = clip(
            input_raster=input_raster,
            input_polygon=input_polygon
        )
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(clipped)
    result = rasterio.open(output_raster_path)

    assert np.amax(result.read()) != np.amin(result.read())
    assert result.count == 1
    # Corresponding operation executed with QGIS for reference solution
    assert result.height == result.width == 26
    assert result.bounds[0] == 384760.0
    assert result.bounds[3] == 6671364.0


def test_clip_wrong_geometry_type():
    """Checks that trying to clip a raster file with non-polygon shape returns custom exception error."""

    with pytest.raises(NotApplicableGeometryTypeException):
        input_point = geopandas.read_file(input_point_path)
        with rasterio.open(input_raster_path) as input_raster:
            clip(
                input_raster=input_raster,
                input_polygon=input_point,
            )


def test_clip_different_crs():
    """Checks that trying to clip a raster file with polygon without matching crs information returns custom exception error."""

    with pytest.raises(NonMatchingCrsException):
        input_polygon = geopandas.read_file(wrong_crs_input_polygon_path)
        with rasterio.open(input_raster_path) as input_raster:
            clip(
                input_raster=input_raster,
                input_polygon=input_polygon,
            )
