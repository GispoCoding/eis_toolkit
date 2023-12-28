import numpy as np


def check_empty_ndarray(array: np.ndarray) -> bool:
    """
    Check if a NumPy array is empty.

    Args:
        array: NumPy array to be checked.
    """
    return array.size == 0
