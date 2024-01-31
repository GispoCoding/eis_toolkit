from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Callable, Optional
from scipy.ndimage import correlate, generic_filter

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidRasterBandException
from eis_toolkit.raster_processing.filters.kernels import _get_kernel_size
from eis_toolkit.utilities.checks.raster import check_single_band


@beartype
def _apply_generic_filter(array: np.ndarray, filter_fn: Callable, kernel: np.ndarray, *args) -> np.ndarray:
    """
    Apply a generic filter to the input array.

    Args:
        array: The input array to be filtered.
        filter_fn: The filter function to be applied.
        kernel: The kernel or footprint to be used for filtering.
        *args: Additional arguments to be passed to the filter function.

    Returns:
        The filtered array.
    """
    return generic_filter(array, filter_fn, footprint=kernel, extra_arguments=args)


@beartype
def _apply_correlated_filter(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a correlated filter to the input array.

    Args:
        array: The input array to be filtered.
        kernel: The filter kernel to be applied.

    Returns:
        The filtered array.
    """
    return correlate(array, kernel) / np.sum(kernel)


@beartype
def _check_filter_size(sigma: Optional[Number], truncate: Optional[Number], size: Optional[int]):
    """
    Check the filter size and raise exceptions if it does not meet the requirements.

    Args:
        sigma: The standard deviation.
        truncate: The truncation value.
        size: The size of the filter kernel.

    Raises:
        InvalidParameterValueException: If the resulting filter radius is too small.
            If the filter size is not allowed.
    """
    if size is None:
        _, radius = _get_kernel_size(sigma, truncate, size)

        if radius < 1:
            raise InvalidParameterValueException(
                "Resulting filter radius too small. Either increase sigma or decrease truncate values."
            )
    else:
        if size < 3:
            raise InvalidParameterValueException("Only numbers larger or equal than 3 are allowed for filter size.")
        elif size % 2 == 0:
            raise InvalidParameterValueException("Only odd numbers are allowed for filter size.")


@beartype
def _check_inputs(
    raster: rasterio.io.DatasetReader,
    size: Optional[int],
    sigma: Optional[Number] = None,
    truncate: Optional[Number] = None,
    **kwargs
):
    """
    Check the inputs.

    Args:
        raster: The input raster.
        size: The size of the filter.
        sigma: The standard deviation.
        truncate: The truncation value.
        **kwargs: Additional keyword arguments.

    Raises:
        InvalidRasterBandException: If the raster has more than one band.
        InvalidParameterValueException: If the value of n_looks is less than 1.
            If the value of damping_factor is negative.
    """
    if check_single_band(raster) is False:
        raise InvalidRasterBandException("Only one-band raster supported.")

    _check_filter_size(sigma, truncate, size)

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            if key == "n_looks" and value < 1:
                raise InvalidParameterValueException(
                    "Only positive numbers larger or equal than 1 are allowed for n_looks."
                )
            elif key == "damping_factor" and value < 0:
                raise InvalidParameterValueException("Only positive numbers are allowed for damping_factor.")
