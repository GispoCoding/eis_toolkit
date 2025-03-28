from contextlib import nullcontext
from functools import partial

import numpy as np
import pytest
import rasterio
import rasterio.plot
import rasterio.profiles
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.exceptions import EmptyDataException, InvalidParameterValueException
from eis_toolkit.raster_processing import distance_to_anomaly
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

with rasterio.open(SMALL_RASTER_PATH) as raster:
    SMALL_RASTER_PROFILE = raster.profile
    SMALL_RASTER_DATA = raster.read(1)

EXPECTED_SMALL_RASTER_SHAPE = SMALL_RASTER_PROFILE["height"], SMALL_RASTER_PROFILE["width"]

EXPECTED_PYTESTPARAMS = [
    pytest.param(
        SMALL_RASTER_PROFILE,
        SMALL_RASTER_DATA,
        5.0,
        "lower",
        EXPECTED_SMALL_RASTER_SHAPE,
        5.6953,
        id="small_raster_lower",
    ),
    pytest.param(
        SMALL_RASTER_PROFILE,
        SMALL_RASTER_DATA,
        5.0,
        "higher",
        EXPECTED_SMALL_RASTER_SHAPE,
        6.4521,
        id="small_raster_higher",
    ),
    pytest.param(
        SMALL_RASTER_PROFILE,
        SMALL_RASTER_DATA,
        (2.5, 7.5),
        "in_between",
        EXPECTED_SMALL_RASTER_SHAPE,
        2.1145,
        id="small_raster_in_between",
    ),
    pytest.param(
        SMALL_RASTER_PROFILE,
        SMALL_RASTER_DATA,
        (2.5, 7.5),
        "outside",
        EXPECTED_SMALL_RASTER_SHAPE,
        32.4904,
        id="small_raster_outside",
    ),
]


def _check_result(out_image, out_profile):
    assert isinstance(out_image, np.ndarray)
    assert isinstance(out_profile, (dict, rasterio.profiles.Profile))
    assert not np.any(np.isnan(out_image))


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
    EXPECTED_PYTESTPARAMS,
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

    # No np.nan expected in input here
    assert not np.any(np.isnan(anomaly_raster_data))
    out_image, out_profile = distance_to_anomaly.distance_to_anomaly(
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria_value=threshold_criteria_value,
        threshold_criteria=threshold_criteria,
    )

    _check_result(out_image=out_image, out_profile=out_profile)

    assert out_image.shape == expected_shape
    if expected_mean is not None:
        assert np.isclose(np.mean(out_image), expected_mean, atol=0.0001)


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
            id="expected_invalid_param_due_to_float_value_in_profile",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "higher",
            partial(dict, transform=None),
            partial(pytest.raises, InvalidParameterValueException),
            id="expected_invalid_param_due_to_none_transform_value",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            5.0,
            "in_between",
            dict,
            partial(pytest.raises, InvalidParameterValueException),
            id="expected_invalid_param_due_to_number_rather_than_range",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            (7.5, 2.5),
            "in_between",
            dict,
            partial(pytest.raises, InvalidParameterValueException),
            id="expected_invalid_param_due_to_invalid_order_in_tuple",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            (1.5, 2.5, 7.5),
            "in_between",
            dict,
            partial(pytest.raises, BeartypeCallHintParamViolation),
            id="expected_invalid_param_due_to_tuple_of_length_three",
        ),
        pytest.param(
            SMALL_RASTER_PROFILE,
            SMALL_RASTER_DATA,
            (100.5, 122.5),
            "in_between",
            dict,
            partial(pytest.raises, EmptyDataException),
            id="expected_empty_data_due_to_threshold_range_outside_values",
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


@pytest.mark.parametrize(
    ",".join(
        [
            "anomaly_raster_profile",
            "anomaly_raster_data",
            "threshold_criteria_value",
            "threshold_criteria",
            "expected_shape",
            "expected_mean_without_nodata",
            "nodata_mask_value",
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
            # Part of values over 5 will now be masked as nodata
            9.0,
            id="small_raster_with_inserted_nodata",
        ),
    ],
)
def test_distance_to_anomaly_nodata_handling(
    anomaly_raster_profile,
    anomaly_raster_data,
    threshold_criteria_value,
    threshold_criteria,
    expected_shape,
    expected_mean_without_nodata,
    nodata_mask_value,
):
    """Test distance_to_anomaly with expected result."""

    anomaly_raster_data_with_nodata = np.where(anomaly_raster_data > nodata_mask_value, np.nan, anomaly_raster_data)
    assert np.any(np.isnan(anomaly_raster_data_with_nodata))

    out_image, out_profile = distance_to_anomaly.distance_to_anomaly(
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data_with_nodata,
        threshold_criteria_value=threshold_criteria_value,
        threshold_criteria=threshold_criteria,
    )

    _check_result(out_image=out_image, out_profile=out_profile)

    assert out_image.shape == expected_shape

    # Result should not be same as without nodata addition
    assert not np.isclose(np.mean(out_image), expected_mean_without_nodata)


# @pytest.mark.parametrize(
#     ",".join(
#         [
#             "anomaly_raster_profile",
#             "anomaly_raster_data",
#             "threshold_criteria_value",
#             "threshold_criteria",
#             "expected_shape",
#             "expected_mean",
#         ]
#     ),
#     EXPECTED_PYTESTPARAMS,
# )
# @pytest.mark.xfail(
#     sys.platform != "win32", reason="gdal_array available only on Windows.", raises=ModuleNotFoundError
# )
# def test_distance_to_anomaly_gdal_expected(
#     anomaly_raster_profile,
#     anomaly_raster_data,
#     threshold_criteria_value,
#     threshold_criteria,
#     expected_shape,
#     expected_mean,
# ):
#     """Test distance_to_anomaly_gdal with expected result."""

#     assert not np.any(np.isnan(anomaly_raster_data))
#     out_image, out_profile = distance_to_anomaly.distance_to_anomaly_gdal(
#         anomaly_raster_profile=anomaly_raster_profile,
#         anomaly_raster_data=anomaly_raster_data,
#         threshold_criteria_value=threshold_criteria_value,
#         threshold_criteria=threshold_criteria,
#     )

#     _check_result(out_image=out_image, out_profile=out_profile)

#     assert out_image.shape == expected_shape
#     if expected_mean is not None:
#         # adding a relative tolerance and absolute tolerance to accomodate the very small error
#         # that arises due to the floating point errors
#         assert np.isclose(np.mean(out_image), expected_mean, rtol=1e-2, atol=1e-2)


# @pytest.mark.parametrize(
#     ",".join(
#         [
#             "anomaly_raster_profile",
#             "anomaly_raster_data",
#             "threshold_criteria_value",
#             "threshold_criteria",
#             "expected_shape",
#             "expected_mean_without_nodata",
#             "nodata_mask_value",
#         ]
#     ),
#     [
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             5.0,
#             "higher",
#             EXPECTED_SMALL_RASTER_SHAPE,
#             6.451948,
#             # Part of values over 5 will now be masked as nodata
#             9.0,
#             id="small_raster_with_inserted_nodata",
#         ),
#     ],
# )
# @pytest.mark.xfail(
#     sys.platform != "win32", reason="gdal_array available only on Windows.", raises=ModuleNotFoundError
# )
# def test_distance_to_anomaly_gdal_nodata_handling(
#     anomaly_raster_profile,
#     anomaly_raster_data,
#     threshold_criteria_value,
#     threshold_criteria,
#     expected_shape,
#     expected_mean_without_nodata,
#     nodata_mask_value,
# ):
#     """Test distance_to_anomaly_gdal with expected result."""

#     anomaly_raster_data_with_nodata = np.where(anomaly_raster_data > nodata_mask_value, np.nan, anomaly_raster_data)
#     assert np.any(np.isnan(anomaly_raster_data_with_nodata))

#     out_image, out_profile = distance_to_anomaly.distance_to_anomaly_gdal(
#         anomaly_raster_profile=anomaly_raster_profile,
#         anomaly_raster_data=anomaly_raster_data_with_nodata,
#         threshold_criteria_value=threshold_criteria_value,
#         threshold_criteria=threshold_criteria,
#     )

#     _check_result(out_image=out_image, out_profile=out_profile)

#     assert out_image.shape == expected_shape

#     # Result should not be same as without nodata addition
#     assert not np.isclose(np.mean(out_image), expected_mean_without_nodata)


# @pytest.mark.parametrize(
#     ",".join(
#         [
#             "anomaly_raster_profile",
#             "anomaly_raster_data",
#             "threshold_criteria_value",
#             "threshold_criteria",
#             "profile_additions",
#             "raises",
#         ]
#     ),
#     [
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             (100.5, 122.5),
#             "in_between",
#             dict,
#             partial(pytest.raises, EmptyDataException),
#             id="expected_empty_data_due_to_threshold_range_outside_values2",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             5.0,
#             "lower",
#             dict,
#             nullcontext,
#             id="no_expected_exception",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             5.0,
#             "higher",
#             partial(dict, height=2.2),
#             partial(pytest.raises, InvalidParameterValueException),
#             id="expected_invalid_param_due_to_float_value_in_profile",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             5.0,
#             "higher",
#             partial(dict, transform=None),
#             partial(pytest.raises, InvalidParameterValueException),
#             id="expected_invalid_param_due_to_none_transform_value",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             5.0,
#             "in_between",
#             dict,
#             partial(pytest.raises, InvalidParameterValueException),
#             id="expected_invalid_param_due_to_number_rather_than_range",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             (7.5, 2.5),
#             "in_between",
#             dict,
#             partial(pytest.raises, InvalidParameterValueException),
#             id="expected_invalid_param_due_to_invalid_order_in_tuple",
#         ),
#         pytest.param(
#             SMALL_RASTER_PROFILE,
#             SMALL_RASTER_DATA,
#             (1.5, 2.5, 7.5),
#             "in_between",
#             dict,
#             partial(pytest.raises, BeartypeCallHintParamViolation),
#             id="expected_invalid_param_due_to_tuple_of_length_three",
#         ),
#     ],
# )
# @pytest.mark.xfail(
#     sys.platform != "win32", reason="gdal_array available only on Windows.", raises=ModuleNotFoundError
# )
# def test_distance_to_anomaly_gdal_expected_check(
#     anomaly_raster_profile,
#     anomaly_raster_data,
#     threshold_criteria_value,
#     threshold_criteria,
#     profile_additions,
#     raises,
# ):
#     """Test distance_to_anomaly_gdal checks."""

#     anomaly_raster_profile.update(profile_additions())
#     anomaly_raster_profile_with_additions = anomaly_raster_profile
#     with raises() as exc_info:
#         out_image, out_profile = distance_to_anomaly.distance_to_anomaly_gdal(
#             anomaly_raster_profile=anomaly_raster_profile_with_additions,
#             anomaly_raster_data=anomaly_raster_data,
#             threshold_criteria_value=threshold_criteria_value,
#             threshold_criteria=threshold_criteria,
#         )

#     if exc_info is not None:
#         # Expected error
#         return

#     _check_result(out_image=out_image, out_profile=out_profile)
