from numbers import Number

import numpy as np
import scipy
from beartype import beartype
from beartype.typing import Literal, Union


@beartype
def _coefficients_horn(
    data: np.ndarray,
    cellsize: Number,
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Calculate the partial derivatives of a surface after after Horn (1981).

    Reference:
        Horn, B.K., 1981: Hill shading and the reflectance map, Proceedings of the IEEE, 69/1: 14-47.

    Args:
        data: The input raster data as a numpy array.
        coefficients: coefficients to calculate.

    Returns:
        The calculated coefficients p, q.
    """
    kernal_p = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernal_q = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (8 * cellsize)
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (8 * cellsize)

    return p, q


@beartype
def _coefficients_zevenbergen(
    data: np.ndarray,
    cellsize: Number,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the partial derivatives of a surface after Zevenbergen & Thorne (1987).

    Reference:
        Zevenbergen, L.W. and Thorne, C.R., 1987: Quantitative analysis of land surface topography,
        Earth Surface Processes and Landforms, 12: 47-56.

    Args:
        data: The input raster data as a numpy array.
        cellsize: The pixel size of the raster data.
        coefficients: coefficients to calculate.

    Returns:
        The calculated coefficients p, q, r, s, t.
    """

    kernal_p = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernal_q = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    kernal_r = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / 2
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / 2

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (2 * cellsize)
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (2 * cellsize)
    r = scipy.ndimage.correlate(data, weights=kernal_r) / (cellsize**2)
    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2)
    t = scipy.ndimage.correlate(data, weights=kernal_t) / (cellsize**2)

    return p, q, r, s, t


@beartype
def _coefficients_young(
    data: np.ndarray,
    cellsize: Number,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the partial derivatives of a surface after the Evans-Young method (1978).

    Reference:
       Young, M., 1978: Terrain analysis program documentation. Report 5 on Grant DA-ERO-591-73-G0040,
       'Statistical characterization of altitude matrices by computer'. Department of Geography,
       University of Durham, England: 27 pp.

       Evans, I.S., 1979: An integrated system of terrain analysis and slope mapping. Report 6 on
       Grant DA-ERO-591-73-G0040, 'Statistical characterization of altitude matrices by computer'.
       Department of Geography, University of Durham, England: 192 pp.

    Args:
        data: The input raster as a numpy array.
        cellsize: The pixel size of the raster data.
        coefficients: coefficients to calculate.

    Returns:
        The calculated coefficients p, q, r, s, t.
    """

    kernal_p = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernal_q = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernal_r = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (6 * cellsize)
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (6 * cellsize)
    r = scipy.ndimage.correlate(data, weights=kernal_r) / (3 * cellsize**2)
    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2)
    t = scipy.ndimage.correlate(data, weights=kernal_t) / (3 * cellsize**2)

    return p, q, r, s, t


@beartype
def _coefficients_evans(
    data: np.ndarray,
    cellsize: Number,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the partial derivatives of a surface after Evans (1979).

    Reference:
       Evans, I.S., 1979: An integrated system of terrain analysis and slope mapping. Report 6 on
       Grant DA-ERO-591-73-G0040, 'Statistical characterization of altitude matrices by computer'.
       Department of Geography, University of Durham, England: 192 pp.

    Args:
        data: The input raster as a numpy array.
        cellsize: The pixel size of the raster data.
        coefficients: coefficients to calculate.

    Returns:
        The calculated coefficients p, q, r, s, t.
    """

    kernal_p = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernal_q = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernal_r_outer = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    kernal_r_inner = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t_outer = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    kernal_t_inner = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (6 * cellsize)
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (6 * cellsize)

    r_outer = scipy.ndimage.correlate(data, weights=kernal_r_outer) / (6 * cellsize**2)
    r_inner = scipy.ndimage.correlate(data, weights=kernal_r_inner) / (3 * cellsize**2)
    r = r_outer - r_inner

    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2)

    t_outer = scipy.ndimage.correlate(data, weights=kernal_t_outer) / (6 * cellsize**2)
    t_inner = scipy.ndimage.correlate(data, weights=kernal_t_inner) / (3 * cellsize**2)
    t = t_outer - t_inner

    return p, q, r, s, t


@beartype
def _coefficients(
    in_array: np.ndarray,
    cellsize: Number,
    method: Literal["Horn", "Evans", "Young", "Zevenbergen"],
) -> tuple[np.ndarray, np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Calculate the partial derivatives of a given surface.

    Args:
        in_array: Input array.
        cellsize: Cellsize of the input raster.
        method: Method to calculate the partial derivatives.

    Returns:
        Tuple of the partial derivatives p, q, r, s, t.
    """
    if method == "Horn":
        p, q = _coefficients_horn(in_array, cellsize)
        return p, q, None, None, None

    if method == "Zevenbergen":
        p, q, r, s, t = _coefficients_zevenbergen(in_array, cellsize)
        return p, q, r, s, t

    if method == "Young":
        p, q, r, s, t = _coefficients_young(in_array, cellsize)
        return p, q, r, s, t

    if method == "Evans":
        p, q, r, s, t = _coefficients_evans(in_array, cellsize)
        return p, q, r, s, t
