from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException
from eis_toolkit.validation.calculate_base_metrics import calculate_base_metrics

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
polygon_path = parent_dir.joinpath("data/remote/small_area.shp")
wrong_crs_polygon_path = parent_dir.joinpath("data/remote/small_area.geojson")


def test_calculate_base_metrics():
    """Tests that the base metrics are calculated correctly."""
    raster = rasterio.open(raster_path)
    deposits = pd.DataFrame({"x": [384829.33, 384829.33], "y": [6671310.87, 6671308.70]})
    deposits = gpd.GeoDataFrame(deposits, geometry=gpd.points_from_xy(deposits.x, deposits.y, crs="EPSG:3067"))

    negatives = pd.DataFrame({"x": [384760.9, 384771.4], "y": [6671319.5, 6671352.1]})
    negatives = gpd.GeoDataFrame(negatives, geometry=gpd.points_from_xy(negatives.x, negatives.y, crs="EPSG:3067"))

    threshold_values = [9.67, 9.55, 2.503]
    true_positive_rate_values = [0.5, 1, 1]
    proportion_of_area_values = [0.00038819875776397513, 0.0015527950310559005, 1]
    false_positive_rate_values = [0.0, 0.0, 1.0]

    reference_metrics = pd.DataFrame(
        {
            "true_positive_rate_values": true_positive_rate_values,
            "proportion_of_area_values": proportion_of_area_values,
            "threshold_values": threshold_values,
            "false_positive_rate_values": false_positive_rate_values,
        }
    )
    metrics = calculate_base_metrics(raster=raster, deposits=deposits, negatives=negatives)

    assert reference_metrics.equals(metrics)


def test_calculate_base_metrics_wrong_geometry_type():
    """Tests that non-polygon geometry raises the correct exception."""
    with pytest.raises(NotApplicableGeometryTypeException):
        polygon = gpd.read_file(polygon_path)
        with rasterio.open(raster_path) as raster:
            calculate_base_metrics(raster=raster, deposits=polygon, negatives=polygon)


def test_calculate_base_metrics_different_crs():
    """Test that a crs mismatch raises the correct exception."""
    with pytest.raises(NonMatchingCrsException):
        wrong_crs_polygon = gpd.read_file(wrong_crs_polygon_path)
        with rasterio.open(raster_path) as raster:
            calculate_base_metrics(
                raster=raster,
                deposits=wrong_crs_polygon,
            )
