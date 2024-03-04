from numbers import Number

import mapclassify as mc
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidRasterBandException
from eis_toolkit.utilities.checks.raster import check_raster_bands


def _reclassify_with_manual_breaks(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    breaks: Sequence[int],
) -> np.ndarray:

    data = np.digitize(band, breaks)

    return data


@beartype
def reclassify_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    breaks: Sequence[int],
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with manual breaks.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        breaks: List of break values for the classification.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with manual breaks.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
    """
    # Add check for input breaks at some point?

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_manual_breaks(band_data, breaks)

    return out_image, out_meta


def _reclassify_with_defined_intervals(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    interval_size: int,
) -> np.ndarray:

    _, edges = np.histogram(band, bins=interval_size)

    data = np.digitize(band, edges)

    return data


@beartype
def reclassify_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    interval_size: int,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with defined intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        interval_size: The number of units in each interval.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with defined intervals.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Interval size is less than 1.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if interval_size < 1:
        raise InvalidParameterValueException("Interval size must be 1 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_defined_intervals(band_data, interval_size)

    return out_image, out_meta


def _reclassify_with_equal_intervals(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_intervals: int,
) -> np.ndarray:

    percentiles = np.linspace(0, 100, number_of_intervals)

    intervals = np.percentile(band, percentiles)

    data = np.digitize(band, intervals)

    return data


@beartype
def reclassify_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with equal intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_intervals: The number of intervals.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with equal intervals.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Number of intervals is less than 2.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if number_of_intervals < 2:
        raise InvalidParameterValueException("Number of intervals must be 2 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_equal_intervals(band_data, number_of_intervals)

    return out_image, out_meta


def _reclassify_with_quantiles(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_quantiles: int,
) -> np.ndarray:

    intervals = [np.percentile(band, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
    data = np.digitize(band, intervals)

    return data


@beartype
def reclassify_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with quantiles.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_quantiles: The number of quantiles.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with quantiles.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Number of quantiles is less than 2.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if number_of_quantiles < 2:
        raise InvalidParameterValueException("Number of quantiles must be 2 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_quantiles(band_data, number_of_quantiles)

    return out_image, out_meta


def _reclassify_with_natural_breaks(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:

    breaks = mc.JenksCaspall(band, number_of_classes)
    data = np.digitize(band, np.sort(breaks.bins))

    return data


@beartype
def reclassify_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with natural breaks (Jenks Caspall).

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_classes: The number of classes.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with natural breaks (Jenks Caspall).
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Number of classes is less than 2.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if number_of_classes < 2:
        raise InvalidParameterValueException("Number of classes must be 2 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_natural_breaks(band_data, number_of_classes)

    return out_image, out_meta


def _reclassify_with_geometrical_intervals(
    band: np.ndarray, number_of_classes: int, nodata_value: Number
) -> np.ndarray:

    # nan_value is either a set integer (e.g. -9999) or np.nan
    mask = band == nodata_value
    masked_array = np.ma.masked_array(data=band, mask=mask)

    median_value = np.ma.median(masked_array)
    max_value = masked_array.max()
    min_value = masked_array.min()

    values_out = np.ma.zeros_like(masked_array)

    # Determine the tail with larger length
    if (median_value - min_value) < (max_value - median_value):  # Large end tail longer
        tail_values = masked_array[np.ma.where((masked_array > median_value))]
        range_tail = max_value - median_value
        tail_values = tail_values - median_value + range_tail / 1000.0
    else:  # Small end tail longer
        tail_values = masked_array[np.ma.where((masked_array < median_value))]
        range_tail = median_value - min_value
        tail_values = tail_values - min_value + range_tail / 1000.0

    min_tail = np.ma.min(tail_values)
    max_tail = np.ma.max(tail_values)

    # number of classes
    factor = (max_tail / min_tail) ** (1 / number_of_classes)

    interval_index = 1
    break_points_tail = [min_tail]
    break_points = [min_tail]
    width = [0]

    while break_points[-1] < max_tail:
        interval_index += 1
        break_points.append(min_tail * factor ** (interval_index - 1))
        break_points_tail.append(break_points[-1])
        width.append(break_points_tail[-1] - break_points_tail[0])
    k = 0

    for j in range(1, len(width) - 2):
        values_out[
            np.ma.where(((median_value + width[j]) < masked_array) & ((masked_array <= (median_value + width[j + 1]))))
        ] = (j + 1)
        values_out[
            np.ma.where(((median_value - width[j]) > masked_array) & ((masked_array >= (median_value - width[j + 1]))))
        ] = (-j - 1)
        k = j

    values_out[np.ma.where(((median_value + width[k + 1]) < masked_array))] = k + 1
    values_out[np.ma.where(((median_value - width[k + 1]) > masked_array))] = -k - 1
    values_out[np.ma.where(median_value == masked_array)] = 0

    output = np.array(values_out)

    return output


@beartype
def reclassify_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, number_of_classes: int, bands: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, dict]:
    """Classify raster with geometrical intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_classes: The number of classes. The true number of classes is at most double the amount,
            depending how symmetrical the input data is.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with geometrical intervals.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Number of classes is less than 2.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if number_of_classes < 2:
        raise InvalidParameterValueException("Number of classes must be 2 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()
    nodata_value = raster.nodata

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_geometrical_intervals(band_data, number_of_classes, nodata_value)

    return out_image, out_meta


def _reclassify_with_standard_deviation(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_intervals: int,
) -> np.ndarray:

    band_statistics = []

    stddev = np.nanstd(band)
    mean = np.nanmean(band)
    band_statistics.append((mean, stddev))
    interval_size = 2 * stddev / number_of_intervals

    classified = np.empty_like(band)

    below_mean = band < (mean - stddev)
    above_mean = band > (mean + stddev)

    classified[below_mean] = -number_of_intervals
    classified[above_mean] = number_of_intervals

    in_between = ~below_mean & ~above_mean
    interval = ((band - (mean - stddev)) / interval_size).astype(int)
    classified[in_between] = interval[in_between] - number_of_intervals // 2

    return classified


@beartype
def reclassify_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Classify raster with standard deviation.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_intervals: The number of intervals.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster data classified with standard deviation.
        Raster metadata.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
        InvalidParameterValueException: Number of intervals is less than 2.
    """

    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    if number_of_intervals < 2:
        raise InvalidParameterValueException("Number of intervals must be 2 or more.")

    out_image = np.empty((len(bands), raster.height, raster.width))
    out_meta = raster.meta.copy()

    for i, band in enumerate(bands):
        band_data = raster.read(band)
        out_image[i] = _reclassify_with_standard_deviation(band_data, number_of_intervals)

    return out_image, out_meta
