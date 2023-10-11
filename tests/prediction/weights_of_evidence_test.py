from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from eis_toolkit import exceptions
from eis_toolkit.prediction.weights_of_evidence import weights_of_evidence_calculate_weights

test_dir = Path(__file__).parent.parent
EVIDENCE_PATH = test_dir.joinpath("../tests/data/remote/wofe/wofe_evidence_raster.tif")
DEPOSIT_PATH = test_dir.joinpath("../tests/data/remote/wofe/wofe_deposits.shp")

evidence_raster = rasterio.open(EVIDENCE_PATH)
deposits = gpd.read_file(DEPOSIT_PATH)


def test_weights_of_evidence():
    """Test that weights of evidence works as intended."""
    df, rasters, raster_meta, _, _ = weights_of_evidence_calculate_weights(evidence_raster, deposits)

    np.testing.assert_equal(df.shape[1], 10)  # 10 columns for unique weights
    np.testing.assert_equal(df.shape[0], 8)  # 8 classes in the test data
    np.testing.assert_equal(len(rasters), 3)  # 3 rasters should be generated with default rasters_to_generate
    np.testing.assert_equal(raster_meta, evidence_raster.meta)


def test_too_high_studentized_contrast_threshold():
    """Tests that too high studentized contrast threshold for reclassification raises the correct exception."""
    with pytest.raises(exceptions.ClassificationFailedException):
        weights_of_evidence_calculate_weights(
            evidence_raster, deposits, weights_type="ascending", studentized_contrast_threshold=2
        )


def test_invalid_choice_in_rasters_to_generate():
    """Tests that invalid metric/column in rasters to generate raises the correct exception."""
    with pytest.raises(exceptions.InvalidColumnException):
        weights_of_evidence_calculate_weights(evidence_raster, deposits, arrays_to_generate=["invalid_metric"])
