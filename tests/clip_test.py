from pathlib import Path
from eis_toolkit.preprocessing.clip import clip_ras
import pytest
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def test_clip():
    """Tests clip functionality with geotiff raster and shapefile polygon."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.shp'
    output_path = '/home/pauliina/Downloads/eis_outputs/clip_result.tif'

    result = clip_ras(input_path, Path(pol_path), output_path)

    # Corresponding operation executed with QGIS for reference solution
    assert result.RasterCount == 1
    band = result.GetRasterBand(1)
    assert result.RasterXSize == 24
    assert result.RasterYSize == 24

    # Compute statistics if needed
    if band.GetMinimum() is None or band.GetMaximum() is None:
        band.ComputeStatistics(0)

    assert band.GetMinimum() == 2.628
    assert band.GetMaximum() == 6.86


def test_clip_wrong_geometry_type():
    """Checks that trying to clip a raster file with non-polygon shape returns custom exception error."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/point.gpkg'
    output_path = '/home/pauliina/Downloads/eis_outputs/clip_result.tif'
    with pytest.raises(NotApplicableGeometryTypeException):
        clip_ras(input_path, Path(pol_path), output_path)


def test_clip_different_crs():
    """Checks that trying to clip a raster file with polygon without matching crs information returns custom exception error."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_df = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.geojson'
    output_path = '/home/pauliina/Downloads/eis_outputs/clip_result.tif'
    with pytest.raises(NonMatchingCrsException):
        clip_ras(input_path, Path(pol_df), output_path)
