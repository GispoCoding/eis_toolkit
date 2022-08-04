import numpy as np
from pathlib import Path
import rasterio
from eis_toolkit.raster_processing.clip import clip_ras
import pytest
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def test_clip():
    """Tests clip functionality with geotiff raster and shapefile polygon."""
    input_path = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif')
    pol_path = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.shp')
    output_path = Path('/home/pauliina/Downloads/eis_outputs/clip_result.tif')

    clip_ras(input_path, pol_path, output_path)

    result = rasterio.open(output_path)

    assert np.amax(result.read()) != np.amin(result.read())
    assert result.count == 1
    # Corresponding operation executed with QGIS for reference solution
    assert result.height == result.width == 26
    assert result.bounds[0] == 384760.0
    assert result.bounds[3] == 6671364.0


def test_clip_wrong_geometry_type():
    """Checks that trying to clip a raster file with non-polygon shape returns custom exception error."""
    input_path = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif')
    pol_path = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/point.gpkg')
    output_path = Path('/home/pauliina/Downloads/eis_outputs/clip_result.tif')
    with pytest.raises(NotApplicableGeometryTypeException):
        clip_ras(input_path, pol_path, output_path)


def test_clip_different_crs():
    """Checks that trying to clip a raster file with polygon without matching crs information returns custom exception error."""
    input_path = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif')
    pol_df = Path('/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.geojson')
    output_path = Path('/home/pauliina/Downloads/eis_outputs/clip_result.tif')
    with pytest.raises(NonMatchingCrsException):
        clip_ras(input_path, pol_df, output_path)
