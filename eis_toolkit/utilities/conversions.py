import numpy as np

from beartype import beartype


@beartype
def convert_rad_to_degree(data: np.ndarray) -> np.ndarray:
    """
    Unit conversion from radians to degree.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in degree values.
    """

    return data * (180.0 / np.pi)


def convert_rad_to_rise(data: np.ndarray) -> np.ndarray:
    """
    Unit conversion from radians to percent rise.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in percent rise values.

    """

    return np.tan(data) * 100.0


def convert_degree_to_rad(data: np.ndarray) -> np.ndarray:
    """
    Unit conversion from degree to radians.

    Args:
      data: Input numpy array.

    Returns:
      The converted array in radian values.
    """

    return (data / 180.0) * np.pi
