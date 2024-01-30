import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.plot

from eis_toolkit.raster_processing import distance_to_anomaly
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

with rasterio.open(SMALL_RASTER_PATH) as raster:
    SMALL_RASTER_PROFILE = raster.profile
    SMALL_RASTER_DATA = raster.read(1)

EXPECTED_SMALL_RASTER_SHAPE = SMALL_RASTER_PROFILE["height"], SMALL_RASTER_PROFILE["width"]


@pytest.mark.parametrize(
    ",".join(
        [
            "raster_profile",
            "anomaly_raster_profile",
            "anomaly_raster_data",
            "threshold_criteria_value",
            "threshold_criteria",
            "expected_shape",
            "expected_mean",
        ]
    ),
    [
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            EXPECTED_SMALL_RASTER_SHAPE,
            6.451948,
            id="small_raster_higher",
        ),
    ],
)
def test_distance_to_anomaly(
    raster_profile,
    anomaly_raster_profile,
    anomaly_raster_data,
    threshold_criteria_value,
    threshold_criteria,
    expected_shape,
    expected_mean,
):
    """Test distance_to_anomaly."""

    result = distance_to_anomaly.distance_to_anomaly(
        raster_profile=raster_profile,
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria_value=threshold_criteria_value,
        threshold_criteria=threshold_criteria,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape
    if expected_mean is not None:
        assert np.isclose(np.mean(result), expected_mean)


@pytest.mark.parametrize(
    ",".join(
        [
            "anomaly_raster_profile",
            "anomaly_raster_data",
            "threshold_criteria_value",
            "threshold_criteria",
            "expected_shape",
            "expected_mean",
        ]
    ),
    [
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            EXPECTED_SMALL_RASTER_SHAPE,
            6.452082,
            id="small_raster_higher",
        ),
    ],
)
@pytest.mark.xfail(
    sys.platform == "win32", reason="GDAL utilities are not available on Windows.", raises=ModuleNotFoundError
)
def test_distance_to_anomaly_gdal(
    anomaly_raster_profile,
    anomaly_raster_data,
    threshold_criteria_value,
    threshold_criteria,
    expected_shape,
    expected_mean,
    tmp_path,
):
    """Test distance_to_anomaly_gdal."""

    output_path = tmp_path / "output.tif"
    result = distance_to_anomaly.distance_to_anomaly_gdal(
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria_value=threshold_criteria_value,
        threshold_criteria=threshold_criteria,
        output_path=output_path,
    )

    assert isinstance(result, Path)
    assert result.is_file()

    with rasterio.open(result) as result_raster:

        assert result_raster.meta["dtype"] in {"float32", "float64"}
        result_raster_data = result_raster.read(1)

    assert result_raster_data.shape == expected_shape
    if expected_mean is not None:
        assert np.isclose(np.mean(result_raster_data), expected_mean)
