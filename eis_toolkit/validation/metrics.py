from typing import Tuple

import geopandas
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString
from sklearn import metrics

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.geometry import check_geometry_types
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def calculate_base_metrics(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    deposits: geopandas.GeoDataFrame,
    band: int = 1,
    negatives: geopandas.GeoDataFrame = None,
    # Lisää negatives ROC-AUC laskemiseen, optionaalinen argumentti
    # Selvitä mistä extra nolla plotissa johtuu
) -> pd.DataFrame:
    """Calculate true positive rate and proportion of area values.

    Function calculates true positive rate and proportion of area values for different thresholds which are
    determined from inputted deposit locations and mineral prospectivity map.

    Args:
        raster (rasterio.io.DatasetReader): Mineral prospectivity map.
        deposits (geopandas.GeoDataFrame): Mineral deposit locations.
        band (int): band index of the mineral prospectivity map, defaults to 1.

    Returns:
        data_frame: data frame containing true positive rate, proportion of area and threshold values.

    Raises:
        NonMatchingCrsException: The raster and polygons are not in the same crs.
        NotApplicableGeometryTypeException: The input geometries contain non-polygon features.
    """
    geometries = deposits["geometry"]

    if not check_matching_crs(
        objects=[raster, geometries],
    ):
        raise NonMatchingCrsException

    if not check_geometry_types(
        geometries=geometries,
        allowed_types=["Point"],
    ):
        raise NotApplicableGeometryTypeException

    data_array = raster.read(band)

    # Select only deposits that are within raster bounds
    deposits = deposits.cx[raster.bounds.left : raster.bounds.right, raster.bounds.bottom : raster.bounds.top]

    deposit_rows, deposit_cols = rasterio.transform.rowcol(raster.transform, deposits.geometry.x, deposits.geometry.y)
    deposit_scores = data_array[deposit_rows, deposit_cols]
    thresholds = np.flip(np.unique(deposit_scores))

    if thresholds.max() < data_array.max():
        thresholds = np.concatenate(([data_array.max()], thresholds))

    # Lisää negatives
    true_positive_rate_values = []
    proportion_of_area_values = []

    for threshold in thresholds:
        true_positive_rate_values.append((deposit_scores >= threshold).sum() / deposit_scores.size)
        proportion_of_area_values.append((data_array >= threshold).sum() / data_array.size)

    if negatives is not None:
        # Select only negatives that are within raster bounds
        negatives = negatives.cx[raster.bounds.left : raster.bounds.right, raster.bounds.bottom : raster.bounds.top]
        negatives_rows, negatives_cols = rasterio.transform.rowcol(
            raster.transform, negatives.geometry.x, negatives.geometry.y
        )
        negatives_scores = data_array[negatives_rows, negatives_cols]

        false_positive_rate_values = []
        for threshold in thresholds:
            false_positive_rate_values.append((negatives_scores >= threshold).sum() / negatives_scores.size)

        false_positive_rate_values.append(1)

    # with 0 threshold, both true positive rate and proportion of area are 1
    true_positive_rate_values.append(1)
    proportion_of_area_values.append(1)
    thresholds = np.append(thresholds, 0)

    data_frame = pd.DataFrame(
        {
            "true_positive_rate": true_positive_rate_values,
            "proportion_of_area": proportion_of_area_values,
            "threshold": thresholds,
        }
    )

    return data_frame


def get_pa_intersection(
    true_positive_rate: np.ndarray, proportion_of_area: np.ndarray, threshold: np.ndarray
) -> Tuple[float, float]:
    """Calculate the intersection point for true positive rate and proportion of area curves.

    Threshold values act as x-axis for both curves. Y-axis for proportion of area is inverted.

    Args:
        true_positive_rate (np.ndarray): True positive rate values.
        proportion_of_area (np.ndarray): Proportion of area values.
        threshold (np.ndarray): Threshold values that were used to calculate true positive rate and proportion of area.

    Returns:
        Tuple[float, float]: x and y coordinates of the intersection point.
    """
    true_positive_area_curve = LineString(np.column_stack((threshold, true_positive_rate)))
    proportion_of_area_curve = LineString(np.column_stack((threshold, 1 - proportion_of_area)))
    intersection = true_positive_area_curve.intersection(proportion_of_area_curve)

    return intersection.x, intersection.y


def calculate_auc(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Calculate area under curve (AUC).

    Calculates AUC for curve where y-axis represents true positive rate and x-axis represents proportion of area.
    AUC is calculated with sklearn.metrics.auc which uses trapezoidal rule for calculation.

    Args:
        true_positive_rate (np.ndarray): _description_
        proportion_of_area (np.ndarray): _description_

    Returns:
        float: _description_
    """
    auc_value = metrics.auc(x_values, y_values)
    return auc_value
