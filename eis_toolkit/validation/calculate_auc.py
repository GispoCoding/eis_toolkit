import numpy as np
from sklearn import metrics

from eis_toolkit.exceptions import InvalidParameterValueException


def _calculate_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    auc_value = metrics.auc(x_values, y_values)

    return float(auc_value)


def calculate_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Calculate area under curve (AUC).

    Calculates AUC for curve. X-axis should be either proportion of area ore false positive rate. Y-axis should be
    always true positive rate. AUC is calculated with sklearn.metrics.auc which uses trapezoidal rule for calculation.

    Args:
        x_values: Either proportion of area or false positive rate values.
        y_values: True positive rate values.

    Returns:
        The area under curve.

    Raises:
        InvalidParameterValueException: x_values or y_values are out of bounds.
    """
    if x_values.max() > 1 or x_values.min() < 0:
        raise InvalidParameterValueException("x_values should be within range 0-1")

    if y_values.max() > 1 or y_values.min() < 0:
        raise InvalidParameterValueException("y_values should be within range 0-1")

    auc_value = _calculate_auc(x_values=x_values, y_values=y_values)
    return auc_value
