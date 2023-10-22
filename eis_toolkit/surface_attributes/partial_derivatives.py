import numpy as np
import rasterio
import scipy
from numbers import Number
from beartype import beartype
from beartype.typing import Literal, Optional, Union, Sequence

from eis_toolkit.utilities.nodata import nodata_to_nan


@beartype
def _method_horn(
    data: np.ndarray,
    cellsize: Number,
    coefficients: Sequence[Literal["p", "q"]],
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Calculate the partial derivatives of a surface after after Horn (1981).

    Reference:
    Horn, B.K., 1981: Hill shading and the reflectance map, Proceedings of the IEEE, 69/1: 14-47.

    Args:
        data: The input raster data as a numpy array.
        coefficients: coefficients to calculate.

    Returns:
        The calculated coefficientss p, q.
    """

    kernal_p = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernal_q = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (8 * cellsize) if "p" in coefficients else None
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (8 * cellsize) if "q" in coefficients else None

    return p, q


@beartype
def _method_zevenbergen_thorne(
    data: np.ndarray,
    cellsize: Number,
    coefficients: Sequence[Literal["p", "q", "r", "s", "t"]],
) -> tuple[
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
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
        The calculated coefficientss p, q, r, s, t.
    """

    kernal_p = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernal_q = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    kernal_r = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / 2
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / 2

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (2 * cellsize) if "p" in coefficients else None
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (2 * cellsize) if "q" in coefficients else None
    r = scipy.ndimage.correlate(data, weights=kernal_r) / (cellsize**2) if "r" in coefficients else None
    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2) if "s" in coefficients else None
    t = scipy.ndimage.correlate(data, weights=kernal_t) / (cellsize**2) if "t" in coefficients else None

    return p, q, r, s, t


@beartype
def _method_evans_young(
    data: np.ndarray,
    cellsize: Number,
    coefficients: Sequence[Literal["p", "q", "r", "s", "t"]],
) -> tuple[
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
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
        The calculated coefficientss p, q, r, s, t.
    """

    kernal_p = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernal_q = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernal_r = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (6 * cellsize) if "p" in coefficients else None
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (6 * cellsize) if "q" in coefficients else None
    r = scipy.ndimage.correlate(data, weights=kernal_r) / (3 * cellsize**2) if "r" in coefficients else None
    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2) if "s" in coefficients else None
    t = scipy.ndimage.correlate(data, weights=kernal_t) / (3 * cellsize**2) if "t" in coefficients else None

    return p, q, r, s, t


@beartype
def _method_evans(
    data: np.ndarray,
    cellsize: Number,
    coefficients: Sequence[Literal["p", "q", "r", "s", "t"]],
) -> tuple[
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
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
        The calculated coefficientss p, q, r, s, t.
    """

    kernal_p = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernal_q = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernal_r_outer = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    kernal_r_inner = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    kernal_s = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    kernal_t_outer = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    kernal_t_inner = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    p = scipy.ndimage.correlate(data, weights=kernal_p) / (6 * cellsize) if "p" in coefficients else None
    q = scipy.ndimage.correlate(data, weights=kernal_q) / (6 * cellsize) if "q" in coefficients else None

    if "r" in coefficients:
        r_outer = scipy.ndimage.correlate(data, weights=kernal_r_outer) / (6 * cellsize**2)
        r_inner = scipy.ndimage.correlate(data, weights=kernal_r_inner) / (3 * cellsize**2)
        r = r_outer - r_inner
    else:
        r = None

    s = scipy.ndimage.correlate(data, weights=kernal_s) / (4 * cellsize**2) if "s" in coefficients else None

    if "t" in coefficients:
        t_outer = scipy.ndimage.correlate(data, weights=kernal_t_outer) / (6 * cellsize**2)
        t_inner = scipy.ndimage.correlate(data, weights=kernal_t_inner) / (3 * cellsize**2)
        t = t_outer - t_inner
    else:
        t = None

    return p, q, r, s, t


@beartype
def coefficients(
    in_array: np.ndarray,
    cellsize: Number,
    method: Literal["Horn", "ZevenbergenThorne", "EvansYoung", "Evans"],
    coefficients: Sequence[Literal["p", "q", "r", "s", "t"]],
) -> tuple[np.ndarray, np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Calculate the partial derivatives of a given surface.

    Args:
        raster: Input raster.
        method: Method to calculate the partial derivatives.
        coefficients: Coefficients to calculate from least squares equation.
        scaling_factor: Factor to modify input raster values, e.g. for unit conversion.

    Returns:
        Tuple of the partial derivatives p, q, r, s, t.
    """

    if method == "Horn":
        p, q = _method_horn(in_array, cellsize, coefficients)
        return p, q, None, None, None

    if method == "ZevenbergenThorne":
        p, q, r, s, t = _method_zevenbergen_thorne(in_array, cellsize, coefficients)
        return p, q, r, s, t

    if method == "EvansYoung":
        p, q, r, s, t = _method_evans_young(in_array, cellsize, coefficients)
        return p, q, r, s, t

    if method == "Evans":
        p, q, r, s, t = _method_evans(in_array, cellsize, coefficients)
        return p, q, r, s, t
