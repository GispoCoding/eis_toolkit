from pathlib import Path
import numpy as np
from eis_toolkit.preprocessing.clip import clip_ras
import pytest
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def test_clip():
    """Tests clip functionality with geotiff raster and shapefile polygon."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.shp'

    result = clip_ras(Path(input_path), pol_path)

    assert np.amax(result[0]) != np.amin(result[0])
    assert result[0].shape[0] == 1
    # Corresponding operation executed with QGIS for reference solution
    assert result[1][2] == 384760.0
    assert result[1][5] == 6671364.0
    print(result)


def test_clip_wrong_geometry_type():
    """Checks that trying to clip a raster file with non-polygon shape returns custom exception error."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/point.gpkg'
    with pytest.raises(NotApplicableGeometryTypeException):
        clip_ras(Path(input_path), pol_path)


def test_clip_different_crs():
    """Checks that trying to clip a raster file with polygon without matching crs information returns custom exception error."""
    input_path = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_raster.tif'
    pol_df = '/home/pauliina/PycharmProjects/eis_toolkit/tests/data/small_area.geojson'
    with pytest.raises(NonMatchingCrsException):
        clip_ras(Path(input_path), pol_df)
