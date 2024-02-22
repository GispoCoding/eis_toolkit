from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional

from eis_toolkit.raster_processing.filters.kernels import _basic_kernel, _gaussian_kernel, _mexican_hat_kernel
from eis_toolkit.raster_processing.filters.utilities import (
    _apply_correlated_filter,
    _apply_generic_filter,
    _check_inputs,
)
from eis_toolkit.utilities.miscellaneous import cast_array_to_float, reduce_ndim
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _focal_median(window: np.ndarray) -> Number:
    """
    Calculate the median value of a window.

    Args:
        window: The filter window.

    Returns:
        The median value of the window.
    """
    weighted_value = np.nanmedian(window) if sum(np.isnan(window)) != len(window) else np.nan
    return weighted_value


@beartype
def focal_filter(
    raster: rasterio.io.DatasetReader,
    method: Literal["mean", "median"] = "mean",
    size: int = 3,
    shape: Literal["square", "circle"] = "circle",
) -> tuple[np.ndarray, dict]:
    """
    Apply a basic focal filter to the input raster.

    Args:
        raster: The input raster dataset.
        method: The method to use for filtering. Can be either "mean" or "median". Default to "mean".
        size: The size of the filter window. E.g., 3 means a 3x3 window. Default to 3.
        shape: The shape of the filter window. Can be either "square" or "circle". Default to "circle".

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
            If the shape is not "square" or "circle".
    """
    _check_inputs(raster, size)

    kernel = _basic_kernel(size, shape)

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    if method == "mean":
        out_array = _apply_correlated_filter(raster_array, kernel)
    elif method == "median":
        out_array = _apply_generic_filter(raster_array, _focal_median, kernel)

    out_array = nan_to_nodata(out_array, raster.nodata)
    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def gaussian_filter(
    raster: rasterio.io.DatasetReader,
    sigma: Number = 1,
    truncate: Number = 4,
    size: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply a gaussian filter to the input raster.

    Args:
        raster: The input raster dataset.
        sigma: The standard deviation of the gaussian kernel.
        truncate: The truncation factor for the gaussian kernel based on the sigma value.
            Only if size is not given. Default to 4.0.
            E.g., for sigma = 1 and truncate = 4.0, the kernel size is 9x9.
        size: The size of the filter window. E.g., 3 means a 3x3 window.
            If size is not None, it overrides the dynamic size calculation based on sigma and truncate.
            Default to None.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
            If the resulting radius is smaller than 1.
    """
    _check_inputs(raster, size, sigma, truncate)

    kernel = _gaussian_kernel(sigma, truncate, size)

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_correlated_filter(raster_array, kernel)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def mexican_hat_filter(
    raster: rasterio.io.DatasetReader,
    sigma: Number = 1,
    truncate: Number = 4,
    size: Optional[int] = None,
    direction: Literal["rectangular", "circular"] = "circular",
) -> tuple[np.ndarray, dict]:
    """
    Apply a mexican hat filter to the input raster.

    Circular: Lowpass filter for smoothing.
    Rectangular: Highpass filter for edge detection. Results may need further normalization.

    Args:
        raster: The input raster dataset.
        sigma: The standard deviation.
        truncate: The truncation factor.
            E.g., for sigma = 1 and truncate = 4.0, the kernel size is 9x9.
            Default to 4.0.
        size: The size of the filter window. E.g., 3 means a 3x3 window. Default to None.
        direction: The direction of calculating the kernel values.
            Can be either "rectangular" or "circular". Default to "circular".

    Returns:
       The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
            If the resulting radius is smaller than 1.
    """
    _check_inputs(raster, size, sigma, truncate)

    kernel = _mexican_hat_kernel(sigma, truncate, size, direction)

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_correlated_filter(raster_array, kernel)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta
