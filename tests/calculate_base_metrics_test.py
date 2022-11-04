from pathlib import Path

import geopandas
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException
from eis_toolkit.validation.calculate_base_metrics import calculate_base_metrics

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
polygon_path = parent_dir.joinpath("data/remote/small_area.shp")
wrong_crs_polygon_path = parent_dir.joinpath("data/remote/small_area.geojson")


def test_calculate_base_metrics_wrong_geometry_type():
    """Tests that non-polygon geometry raises the correct exception."""
    with pytest.raises(NotApplicableGeometryTypeException):
        polygon = geopandas.read_file(polygon_path)
        with rasterio.open(raster_path) as raster:
            calculate_base_metrics(raster=raster, deposits=polygon, negatives=polygon)


def test_calculate_base_metrics_different_crs():
    """Test that a crs mismatch raises the correct exception."""
    with pytest.raises(NonMatchingCrsException):
        wrong_crs_polygon = geopandas.read_file(wrong_crs_polygon_path)
        with rasterio.open(raster_path) as raster:
            calculate_base_metrics(
                raster=raster,
                deposits=wrong_crs_polygon,
            )
