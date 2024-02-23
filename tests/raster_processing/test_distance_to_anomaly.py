import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.plot
import rasterio.profiles

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing import distance_to_anomaly
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

with rasterio.open(SMALL_RASTER_PATH) as raster:
    SMALL_RASTER_PROFILE = raster.profile
    SMALL_RASTER_DATA = raster.read(1)

EXPECTED_SMALL_RASTER_SHAPE = SMALL_RASTER_PROFILE["height"], SMALL_RASTER_PROFILE["width"]


def _check_result(out_image, out_profile):
    assert isinstance(out_image, np.ndarray)
    assert isinstance(out_profile, (dict, rasterio.profiles.Profile))


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
            6.451948,
            id="small_raster_higher",
        ),
    ],
)
def test_distance_to_anomaly_expected(
    anomaly_raster_profile,
    anomaly_raster_data,
    threshold_criteria_value,
    threshold_criteria,
    expected_shape,
    expected_mean,
):
    """Test distance_to_anomaly with expected result."""

    out_image, out_profile = distance_to_anomaly.distance_to_anomaly(
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria_value=threshold_criteria_value,
        threshold_criteria=threshold_criteria,
    )

    _check_result(out_image=out_image, out_profile=out_profile)

    assert out_image.shape == expected_shape
    if expected_mean is not None:
        assert np.isclose(np.mean(out_image), expected_mean)


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


@pytest.mark.parametrize(
    ",".join(
        [
            "anomaly_raster_profile",
            "anomaly_raster_data",
            "threshold_criteria_value",
            "threshold_criteria",
            "profile_additions",
            "raises",
        ]
    ),
    [
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            dict,
            nullcontext,
            id="no_expected_exception",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            partial(dict, height=2.2),
            partial(pytest.raises, InvalidParameterValueException),
            id="expected_invalid_param_due_to_float_value",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            partial(dict, transform=None),
            partial(pytest.raises, InvalidParameterValueException),
            id="expected_invalid_param_due_to_transform_value",
        ),
    ],
)
def test_distance_to_anomaly_check(
    anomaly_raster_profile,
    anomaly_raster_data,
    threshold_criteria_value,
    threshold_criteria,
    profile_additions,
    raises,
):
    """Test distance_to_anomaly checks."""

    anomaly_raster_profile_with_additions = {**anomaly_raster_profile, **profile_additions()}
    with raises() as exc_info:
        out_image, out_profile = distance_to_anomaly.distance_to_anomaly(
            anomaly_raster_profile=anomaly_raster_profile_with_additions,
            anomaly_raster_data=anomaly_raster_data,
            threshold_criteria_value=threshold_criteria_value,
            threshold_criteria=threshold_criteria,
        )

    if exc_info is not None:
        # Expected error
        return

    _check_result(out_image=out_image, out_profile=out_profile)
