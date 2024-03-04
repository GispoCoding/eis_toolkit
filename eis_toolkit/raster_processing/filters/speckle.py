from numbers import Number

import numpy as np
import rasterio
from beartype import beartype

from eis_toolkit.raster_processing.filters.utilities import _apply_generic_filter, _check_inputs
from eis_toolkit.utilities.miscellaneous import cast_array_to_float, reduce_ndim
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _lee_additive_noise(window: np.ndarray, add_noise_var: Number) -> Number:
    """
    Calculate the weighted value for a Lee filter (additive noise) from a window of pixels.

    Args:
        window: The moving window of pixels.
        add_noise_var: The variance of the additive noise.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_var = np.nanvar(window)
        local_mean = np.nanmean(window)

        weight = local_var / (local_var + add_noise_var)
        weighted_value = local_mean + weight * (p_center - local_mean)
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def _lee_multiplicative_noise(window: np.ndarray, mult_noise_mean: Number, n_looks: int) -> Number:
    """
    Calculate the weighted value for a Lee filter (multiplicative noise) from a window of pixels.

    Args:
        window: The moving window of pixels.
        mult_noise_mean: The mean of the multiplicative noise.
        n_looks: Number of looks to estimate the noise variation.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_var = np.nanvar(window)
        local_mean = np.nanmean(window)

        mult_noise_var = 1 / n_looks

        numerator = mult_noise_mean * local_var
        denumerator = (mult_noise_var * local_mean**2) + (local_var * mult_noise_mean**2)

        weight = numerator / denumerator if (numerator != 0 and denumerator != 0) else 0
        weighted_value = local_mean + weight * (p_center - (mult_noise_mean * local_mean))
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def _lee_additive_multiplicative_noise(
    window: np.ndarray, add_noise_var: Number, add_noise_mean: Number, mult_noise_mean: Number
) -> Number:
    """
    Calculate the weighted value for a Lee filter (additive and multiplicative noise) from a window of pixels.

    Args:
        window: The moving window of pixels.
        add_noise_var: The variance of the additive noise.
        add_noise_mean: The mean of the additive noise.
        mult_noise_mean: The mean of the multiplicative noise.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_var = np.nanvar(window)
        local_sd = np.nanstd(window)
        local_mean = np.nanmean(window)

        mult_noise_var = np.power((local_sd / local_mean), 2) if (local_sd != 0 and local_mean != 0) else 0
        weight = (mult_noise_mean * local_var) / (
            (mult_noise_var * local_mean**2) + (local_var * mult_noise_mean**2) + add_noise_var
        )
        weighted_value = local_mean + weight * (p_center - (mult_noise_mean * local_mean) - add_noise_mean)
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def _lee_enhanced(window: np.ndarray, n_looks: int, damping_factor: Number) -> Number:
    """
    Calculate the weighted value for a Lee enhanced filter from a window of pixels.

    Args:
        window: The moving window of pixels.
        n_looks: Number of looks to estimate the noise variation.
        damping_factor: Damping effect on filtering.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_sd = np.nanstd(window)
        local_mean = np.nanmean(window)

        noise_sd = np.sqrt(1 / n_looks)
        variation = local_sd / local_mean if (local_sd != 0 and local_mean != 0) else 0
        noise_sd_max = np.sqrt(1 + 2 / n_looks)

        exponent = (
            -damping_factor * (variation - noise_sd) / (noise_sd_max - variation) if noise_sd_max != variation else 0
        )
        weight = np.exp(exponent)

        if variation <= noise_sd:
            weighted_value = local_mean
        elif noise_sd < variation < noise_sd_max:
            weighted_value = (local_mean * weight) + p_center * (1 - weight)
        elif variation >= noise_sd_max:
            weighted_value = p_center
    else:
        weighted_value = np.nan

    return weighted_value


def _gamma(window: np.ndarray, n_looks: int) -> np.ndarray:
    """
    Calculate the weighted value for a Gamma filter from a window of pixels.

    Args:
        window: The moving window of pixels.
        n_looks: Number of looks to estimate the noise variation.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_sd = np.nanstd(window)
        local_mean = np.nanmean(window)

        noise_sd = np.sqrt(1 / n_looks)
        variation = local_sd / local_mean if (local_sd != 0 and local_mean != 0) else 0
        noise_sd_max = np.sqrt(2) * noise_sd

        factor_a = (1 + noise_sd**2) / (variation**2 - noise_sd**2)
        factor_b = factor_a - (n_looks - 1)
        factor_d = local_mean**2 * factor_b**2 + 4 * factor_a * n_looks * local_mean * p_center

        if variation <= noise_sd:
            weighted_value = local_mean
        elif noise_sd < variation < noise_sd_max:
            weighted_value = (factor_b * local_mean + np.sqrt(factor_d)) / (2 * factor_a)
        elif variation >= noise_sd_max:
            weighted_value = p_center
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def _frost(window: np.ndarray, damping_factor: Number) -> Number:
    """
    Calculate the weighted value for a Frost filter from a window of pixels.

    Args:
        window: The moving window of pixels.
        damping_factor: Damping effect on filtering.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        s_dist = np.abs(window - p_center)
        local_var = np.nanvar(window)
        local_mean = np.nanmean(window)

        scaled_var = local_var / local_mean**2 if (local_var != 0 and local_mean != 0) else 0
        factor_b = damping_factor * scaled_var
        array_weights = np.exp(-factor_b * s_dist)

        weighted_array = window * array_weights
        weighted_value = np.nansum(weighted_array) / np.nansum(array_weights)
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def _kuan(window: np.ndarray, n_looks: int) -> Number:
    """
    Calculate the weighted value for a Kuan filter from a window of pixels.

    Args:
        window: The moving window of pixels.
        n_looks: Number of looks to estimate the noise variation.

    Returns:
        The filtered value.
    """
    p_center = window[window.shape[0] // 2]

    if not np.isnan(p_center):
        local_sd = np.nanstd(window)
        local_mean = np.nanmean(window)

        noise_sd = np.sqrt(1 / n_looks)
        variation = local_sd / local_mean if (local_sd != 0 and local_mean != 0) else 0

        if variation <= noise_sd:
            weight = 0
        else:
            weight = (1 - (noise_sd**2 / variation**2)) / (1 + noise_sd**2) if variation != 0 else 0

        weighted_value = (p_center * weight) + local_mean * (1 - weight)
    else:
        weighted_value = np.nan

    return weighted_value


@beartype
def lee_additive_noise_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    add_noise_var: Number = 0.25,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Lee filter considering additive noise components in the input raster.

    Lower noise values result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        add_noise_var: The additive noise variation. Default to 0.25.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)

    raster_array = nodata_to_nan(raster_array, raster.nodata)
    out_array = _apply_generic_filter(raster_array, _lee_additive_noise, kernel, add_noise_var)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def lee_multiplicative_noise_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    mult_noise_mean: Number = 1,
    n_looks: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Lee filter considering multiplicative noise components in the input raster.

    Higher number of looks result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        mult_noise_mean: The multiplative noise mean. Default to 1.
        n_looks: Number of looks to estimate the noise variation.
            Higher values result in higher smoothing. Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size, n_looks=n_looks)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(raster_array, _lee_multiplicative_noise, kernel, mult_noise_mean, n_looks)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def lee_additive_multiplicative_noise_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    add_noise_var: Number = 0.25,
    add_noise_mean: Number = 0,
    mult_noise_mean: Number = 1,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Lee filter considering both additive and multiplicative noise components in the input raster.

    Lower noise values result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        add_noise_var: The additive noise variation. Default to 0.25.
        add_noise_mean: The additive noise mean. Default to 0.
        mult_noise_mean: The multiplative noise mean. Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(
        raster_array, _lee_additive_multiplicative_noise, kernel, add_noise_var, add_noise_mean, mult_noise_mean
    )

    out_array = nan_to_nodata(out_array, raster.nodata)
    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def lee_enhanced_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    n_looks: int = 1,
    damping_factor: Number = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Apply an enhanced Lee filter to the input raster.

    Higher number of looks and damping factor result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        n_looks: Number of looks to estimate the noise variation.
            Higher values result in higher smoothing.
            Low values may result in focal mean filtering.
            Default to 1.
        damping_factor: Extent of exponential damping effect on filtering.
            Larger damping values preserve edges better but smooths less.
            Smaller values produce more smoothing.
            Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size, n_looks=n_looks, damping_factor=damping_factor)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(raster_array, _lee_enhanced, kernel, n_looks, damping_factor)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def gamma_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    n_looks: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Gamma filter to the input raster.

    Higher number of looks result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        n_looks: Number of looks to estimate the noise variation.
            Higher values result in higher smoothing.
            Low values may result in focal mean filtering.
            Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size, n_looks=n_looks)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(raster_array, _gamma, kernel, n_looks)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def frost_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    damping_factor: Number = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Frost filter to the input raster.

    Higher damping factor result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        damping_factor: Extent of exponential damping effect on filtering.
            Larger damping values preserve edges better but smooths less.
            Smaller values produce more smoothing.
            Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size, damping_factor=damping_factor)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(raster_array, _frost, kernel, damping_factor)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta


@beartype
def kuan_filter(
    raster: rasterio.io.DatasetReader,
    size: int = 3,
    n_looks: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Apply a Kuan filter to the input raster.

    Higher number of looks result in better edge preservation.

    Args:
        raster: The input raster dataset.
        size: The size of the filter window.
            E.g., 3 means a 3x3 window. Default to 3.
        n_looks: Number of looks to estimate the noise variation.
            Higher values result in higher smoothing.
            Low values may result in focal mean filtering.
            Default to 1.

    Returns:
        The filtered raster array.

    Raises:
        InvalidRasterBandException: If the input raster has more than one band.
        InvalidParameterValueException: If the filter size is smaller than 3.
            If the filter size is not an odd number.
    """
    _check_inputs(raster, size, n_looks=n_looks)

    kernel = np.ones((size, size))

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, raster.nodata)

    out_array = _apply_generic_filter(raster_array, _kuan, kernel, n_looks)
    out_array = nan_to_nodata(out_array, raster.nodata)

    out_array = cast_array_to_float(out_array, cast_float=True)
    out_meta = raster.meta.copy()

    return out_array, out_meta
