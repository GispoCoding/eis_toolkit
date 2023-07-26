import scipy
import numpy as np

from beartype import beartype
from beartype import Literal
from numbers import Number


@beartype
def _method_horn(
    data: np.ndarray,
    cellsize: Number,
    parameter: Literal,
) -> np.ndarray:
    """
    Calculate the partial derivatives of a surface after after Horn (1981).

    Source:
    Horn, B.K., 1981: Hill shading and the reflectance map, Proceedings of the IEEE, 69/1: 14-47.

    Args:
        data: The input raster data as a numpy array.
        parameter: Parameter to calculate.

    Returns:
        The calculated parameter p or q.
    """

    kernal_p = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernal_q = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if parameter == "p":
        out_array = scipy.ndimage.correlate(data, weights=kernal_p) / (8 * cellsize)
    elif parameter == "q":
        out_array = scipy.ndimage.correlate(data, weights=kernal_q) / (8 * cellsize)

    return out_array


@beartype
def _method_zevenbergen(
    data: np.ndarray,
    cellsize: Number,
    parameter: Literal,
) -> np.ndarray:
    """
    Calculate the partial derivatives of a surface after Zevenbergen & Thorne (1987).

    Source:
    Zevenbergen, L.W. and Thorne, C.R., 1987: Quantitative analysis of land surface topography, Earth Surface Processes and Landforms, 12: 47-56.

    Args:
        data: The input raster data as a numpy array.
        parameter: Parameter to calculate.

    Returns:
        The calculated parameter p, q, r, s or t.
    """

    kernal_p = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernal_q = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    kernal_r = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / 2
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / 2

    if parameter == "p":
        out_array = scipy.ndimage.correlate(data, weights=kernal_p) / (2 * cellsize)
    elif parameter == "q":
        out_array = scipy.ndimage.correlate(data, weights=kernal_q) / (2 * cellsize)
    elif parameter == "r":
        out_array = scipy.ndimage.correlate(data, weights=kernal_r) / (cellsize**2)
    elif parameter == "s":
        out_array = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2)
    elif parameter == "t":
        out_array = scipy.ndimage.correlate(data, weights=kernal_t) / (cellsize**2)

    return out_array
