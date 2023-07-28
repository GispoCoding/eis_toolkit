import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from shapely.geometry import LineString
from shapely.geometry.point import Point

from eis_toolkit.exceptions import InvalidParameterValueException


def _get_pa_intersection(
    true_positive_rate_values: np.ndarray, proportion_of_area_values: np.ndarray, threshold_values: np.ndarray
) -> Point:
    true_positive_area_curve = LineString(np.column_stack((threshold_values, true_positive_rate_values)))
    proportion_of_area_values_curve = LineString(np.column_stack((threshold_values, 1 - proportion_of_area_values)))
    intersection = true_positive_area_curve.intersection(proportion_of_area_values_curve)

    return intersection


@beartype
def get_pa_intersection(
    true_positive_rate_values: np.ndarray, proportion_of_area_values: np.ndarray, threshold_values: np.ndarray
) -> Tuple[float, float]:
    """Calculate the intersection point for prediction rate and area curves in (P-A plot).

    Threshold_values values act as x-axis for both curves. Prediction rate curve uses true positive rate for y-axis.
    Area curve uses inverted proportion of area as y-axis.

    Args:
        true_positive_rate_values: True positive rate values, values should be within range 0-1.
        proportion_of_area_values: Proportion of area values, values should be within range 0-1.
        threshold_values: Threshold values that were used to calculate true positive rate and proportion of
        area.

    Returns:
        X and y coordinates of the intersection point.

    Raises:
        InvalidParameterValueException: true_positive_rate_values or proportion_of_area_values values are out of bounds.
    """
    if true_positive_rate_values.max() > 1 or true_positive_rate_values.min() < 0:
        raise InvalidParameterValueException("true_positive_rate_values values should be within range 0-1")

    if proportion_of_area_values.max() > 1 or proportion_of_area_values.min() < 0:
        raise InvalidParameterValueException("proportion_of_area_values values should be within range 0-1")

    intersection = _get_pa_intersection(
        true_positive_rate_values=true_positive_rate_values,
        proportion_of_area_values=proportion_of_area_values,
        threshold_values=threshold_values,
    )

    return intersection.x, intersection.y
