from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.prediction.weights_of_evidence import (
    CLASS_COLUMN,
    CONTRAST_COLUMN,
    GENERALIZED_CLASS_COLUMN,
    S_CONTRAST_COLUMN,
    STUDENTIZED_CONTRAST_COLUMN,
    WEIGHT_MINUS_COLUMN,
    WEIGHT_PLUS_COLUMN,
    WEIGHT_S_MINUS_COLUMN,
    WEIGHT_S_PLUS_COLUMN,
    generalize_weights_cumulative,
    weights_of_evidence_calculate_responses,
    weights_of_evidence_calculate_weights,
)
from eis_toolkit.warnings import ClassificationFailedWarning

test_dir = Path(__file__).parent.parent
EVIDENCE_PATH = test_dir.joinpath("../tests/data/remote/wofe/wofe_evidence_raster.tif")
DEPOSIT_PATH = test_dir.joinpath("../tests/data/remote/wofe/wofe_deposits.shp")

evidence_raster = rasterio.open(EVIDENCE_PATH)
deposits = gpd.read_file(DEPOSIT_PATH)

weights_table = pd.DataFrame(
    [
        [1.0, 0.5059, 0.4083, -0.0877, 0.1961, 0.5936, 0.4529, 1.3106],
        [2.0, 0.4387, 0.3015, -0.1706, 0.2182, 0.6093, 0.3722, 1.6371],
        [3.0, 0.3942, 0.2426, -0.3142, 0.2582, 0.7084, 0.3543, 1.9997],
        [4.0, 0.2571, 0.2236, -0.3205, 0.2887, 0.5777, 0.3652, 1.5819],
        [5.0, 0.2877, 0.1961, -0.7339, 0.4083, 1.0216, 0.4529, 2.2555],
        [6.0, 0.2683, 0.1826, -1.5107, 0.7071, 1.7790, 0.7303, 2.4360],
        [7.0, 0.1722, 0.1796, -1.7844, 1.0050, 1.9566, 1.0210, 1.9164],
        [8.0, 0.0608, 0.1796, -1.0464, 1.0051, 1.1072, 1.0210, 1.0844],
        [9.0, -0.0314, 0.1796, 9.4195, 1.4213, -9.4509, 1.4326, -6.5968],
    ],
    columns=[
        CLASS_COLUMN,
        WEIGHT_PLUS_COLUMN,
        WEIGHT_S_PLUS_COLUMN,
        WEIGHT_MINUS_COLUMN,
        WEIGHT_S_MINUS_COLUMN,
        CONTRAST_COLUMN,
        S_CONTRAST_COLUMN,
        STUDENTIZED_CONTRAST_COLUMN,
    ],
)


def test_weights_of_evidence():
    """Test that weights of evidence works as intended."""
    df, rasters, raster_meta, _, _ = weights_of_evidence_calculate_weights(evidence_raster, deposits)

    np.testing.assert_equal(df.shape[1], 10)  # 10 columns for unique weights
    np.testing.assert_equal(df.shape[0], 8)  # 8 classes in the test data
    np.testing.assert_equal(len(rasters), 3)  # 3 rasters should be generated with default rasters_to_generate
    np.testing.assert_equal(raster_meta, evidence_raster.meta)


def test_too_high_studentized_contrast_threshold():
    """Tests that too high studentized contrast threshold for reclassification results in warning."""
    with pytest.warns(ClassificationFailedWarning):
        result, _, _, _, _ = weights_of_evidence_calculate_weights(
            evidence_raster, deposits, weights_type="ascending", studentized_contrast_threshold=2
        )

        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_invalid_choice_in_rasters_to_generate():
    """Tests that invalid metric/column in rasters to generate raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        weights_of_evidence_calculate_weights(evidence_raster, deposits, arrays_to_generate=["invalid_metric"])


def test_cumulative_reclassification_manual():
    """Test generalizing cumulative weights with the 'manual' option."""
    df = weights_table.copy()

    result = generalize_weights_cumulative(df, "manual", manual_cutoff_index=4)

    assert GENERALIZED_CLASS_COLUMN in result.columns.values

    expected_output = pd.Series([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=np.int64)
    pd.testing.assert_series_equal(expected_output, result[GENERALIZED_CLASS_COLUMN], check_names=False)

    with pytest.warns(ClassificationFailedWarning):
        result = generalize_weights_cumulative(df, "manual", 8)
        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_cumulative_reclassification_max_contrast():
    """Test generalizing cumulative weights with the 'max_contrast' option."""
    df = weights_table.copy()

    result = generalize_weights_cumulative(df, "max_contrast")

    assert GENERALIZED_CLASS_COLUMN in result.columns.values

    expected_output = pd.Series([2, 2, 2, 2, 2, 2, 2, 1, 1], dtype=np.int64)
    pd.testing.assert_series_equal(expected_output, result[GENERALIZED_CLASS_COLUMN], check_names=False)

    with pytest.warns(ClassificationFailedWarning):
        df = weights_table.copy()
        df = df.head(3)  # Last row has the highest contrast

        result = generalize_weights_cumulative(df, "max_contrast")

        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_cumulative_reclassification_max_contrast_if_feasible():
    """Test generalizing cumulative weights with the 'max_contrast_if_feasible' option."""
    df = weights_table.copy()

    result = generalize_weights_cumulative(df, "max_contrast_if_feasible", studentized_contrast_threshold=1)

    assert GENERALIZED_CLASS_COLUMN in result.columns.values

    expected_output = pd.Series([2, 2, 2, 2, 2, 2, 2, 1, 1], dtype=np.int64)
    pd.testing.assert_series_equal(expected_output, result[GENERALIZED_CLASS_COLUMN], check_names=False)

    with pytest.warns(ClassificationFailedWarning):
        result = generalize_weights_cumulative(df, "max_contrast_if_feasible", studentized_contrast_threshold=2)
        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_cumulative_reclassification_max_feasible_contrast():
    """Test generalizing cumulative weights with the 'max_feasible_contrast' option."""
    df = weights_table.copy()

    result = generalize_weights_cumulative(df, "max_feasible_contrast", studentized_contrast_threshold=2)

    assert GENERALIZED_CLASS_COLUMN in result.columns.values

    expected_output = pd.Series([2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.int64)
    pd.testing.assert_series_equal(expected_output, result[GENERALIZED_CLASS_COLUMN], check_names=False)

    with pytest.warns(ClassificationFailedWarning):
        result = generalize_weights_cumulative(df, "max_feasible_contrast", studentized_contrast_threshold=2.5)
        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_cumulative_reclassification_max_studentized_contrast():
    """Test generalizing cumulative weights with the 'max_studentized_contrast' option."""
    df = weights_table.copy()

    result = generalize_weights_cumulative(df, "max_studentized_contrast")

    assert GENERALIZED_CLASS_COLUMN in result.columns.values

    expected_output = pd.Series([2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.int64)
    pd.testing.assert_series_equal(expected_output, result[GENERALIZED_CLASS_COLUMN], check_names=False)

    with pytest.warns(ClassificationFailedWarning):
        df = weights_table.copy()
        df = df.head(3)  # Last row has the highest studentized contrast

        result = generalize_weights_cumulative(df, "max_studentized_contrast")
        assert GENERALIZED_CLASS_COLUMN not in result.columns.values


def test_calculate_responses_invalid_nr_of_deposits():
    """Tests that an exception is raised if nr_of_deposits > nr_of_pixels or either value is not positive."""
    df = weights_table.copy()
    output_arrays = [
        {WEIGHT_PLUS_COLUMN: df[WEIGHT_PLUS_COLUMN].to_numpy()},
        {WEIGHT_S_PLUS_COLUMN: df[WEIGHT_S_PLUS_COLUMN].to_numpy()},
    ]
    with pytest.raises(InvalidParameterValueException):
        weights_of_evidence_calculate_responses(output_arrays, nr_of_deposits=2, nr_of_pixels=1)

    with pytest.raises(InvalidParameterValueException):
        weights_of_evidence_calculate_responses(output_arrays, nr_of_deposits=1, nr_of_pixels=0)

    with pytest.raises(InvalidParameterValueException):
        weights_of_evidence_calculate_responses(output_arrays, nr_of_deposits=-1, nr_of_pixels=1)
