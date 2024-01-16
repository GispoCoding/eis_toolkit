import numpy as np
from beartype import beartype


@beartype
def convert_rad_to_deg(data: np.ndarray) -> np.ndarray:
    """
    Convert radians to degree.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in degree values.
    """

    return np.where(data >= 0, np.degrees(data), data)


@beartype
def convert_rad_to_rise(data: np.ndarray) -> np.ndarray:
    """
    Convert radians to percent rise.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in percent rise values.
    """

    return np.where(data >= 0, np.tan(data) * 100.0, data)


@beartype
def convert_deg_to_rise(data: np.ndarray) -> np.ndarray:
    """
    Convert degrees to percent rise.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in percent rise values.
    """

    return np.where(data >= 0, np.tan(np.radians(data)) * 100.0, data)


@beartype
def convert_deg_to_rad(data: np.ndarray) -> np.ndarray:
    """
    Convert degree to radians.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in radian values.
    """

    return np.where(data >= 0, np.radians(data), data)


@beartype
def convert_rise_to_deg(data: np.ndarray) -> np.ndarray:
    """
    Convert percent rise to degrees.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in degree.
    """

    return np.where(data >= 0, np.degrees(np.arctan(data / 100)), data)
