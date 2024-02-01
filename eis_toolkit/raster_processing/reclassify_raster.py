import mapclassify as mc
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Sequence, Tuple, Union

from eis_toolkit.exceptions import InvalidParameterValueException


def _bands_non_negative(band: list):
    if any(n < 0 for n in band):
        raise InvalidParameterValueException("The list bands contains negative values.")


def _raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    breaks: Sequence[int],
) -> np.ndarray:

    out_image = []

    data = np.digitize(band, breaks)

    out_image.append(data)

    return out_image


@beartype
def raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    breaks: Sequence[int],
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with manual breaks.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        breaks: List of break values for the classification.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with manual breaks and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        manual_breaks_band = _raster_with_manual_breaks(breaks, band)

        out_image.append(manual_breaks_band)

    return out_image, out_meta


def _raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    interval_size: int,
) -> np.ndarray:

    out_image = []

    _, edges = np.histogram(band, bins=interval_size)

    data = np.digitize(band, edges)

    out_image.append(data)

    return out_image


@beartype
def raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    interval_size: int,
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with defined intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        interval_size: The number of units in each interval.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with defined intervals and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        defined_intervals_band = _raster_with_defined_intervals(band, interval_size)

        out_image.append(defined_intervals_band)

    return out_image, out_meta


def _raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_intervals: int,
) -> np.ndarray:

    out_image = []

    percentiles = np.linspace(0, 100, number_of_intervals)
    intervals = np.percentile(band, percentiles)
    data = np.digitize(band, intervals)
    out_image.append(data)

    return out_image


@beartype
def raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with equal intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_intervals: The number of intervals.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with equal intervals.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        equal_intervals_band = _raster_with_equal_intervals(band, number_of_intervals)

        out_image.append(equal_intervals_band)

    return out_image, out_meta


def _raster_with_quantiles(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_quantiles: int,
) -> np.ndarray:

    out_image = []

    intervals = [np.percentile(band, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
    data = np.digitize(band, intervals)

    out_image.append(data)

    return out_image


@beartype
def raster_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with quantiles.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_quantiles: The number of quantiles.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with quantiles and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        numbered_quantiles_band = _raster_with_quantiles(band, number_of_quantiles)

        out_image.append(numbered_quantiles_band)

    return out_image, out_meta


def _raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:

    out_image = []

    breaks = mc.JenksCaspall(band, number_of_classes)
    data = np.digitize(band, np.sort(breaks.bins))

    out_image.append(data)

    return out_image


@beartype
def raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with natural breaks (Jenks Caspall).

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_classes: The number of classes.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with natural breaks (Jenks Caspall) and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        natural_breaks_band = _raster_with_natural_breaks(band, number_of_classes)

        out_image.append(natural_breaks_band)

    return out_image, out_meta


def _raster_with_geometrical_intervals(
    band: np.ndarray, number_of_classes: int, nan_value: Union[int, float]
) -> np.ndarray:

    out_image = []
    is_integer = np.issubdtype(band.dtype, int)
    is_float = np.issubdtype(band.dtype, float)

    if is_integer:
        mask = band == nan_value
        band[mask] = nan_value
    elif is_float:
        band[band == nan_value] = np.nan

    median_value = np.nanmedian(band)
    max_value = np.nanmax(band)
    min_value = np.nanmin(band)

    values_out = np.zeros_like(band)

    nan_condition = ~np.isnan(band)

    # Determine the tail with larger length
    if (median_value - min_value) < (max_value - median_value):  # Large end tail longer
        tail_values = band[np.where((band > median_value) & nan_condition)]
        range_tail = max_value - median_value
        tail_values = tail_values - median_value + range_tail / 1000.0
    else:  # Small end tail longer
        tail_values = band[np.where((band < median_value) & nan_condition)]
        range_tail = median_value - min_value
        tail_values = tail_values - min_value + range_tail / 1000.0

    min_tail = np.nanmin(tail_values)
    max_tail = np.nanmax(tail_values)

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
            np.where(((median_value + width[j]) < band) & ((band <= (median_value + width[j + 1])) & nan_condition))
        ] = (j + 1)
        values_out[
            np.where(((median_value - width[j]) > band) & ((band >= (median_value - width[j + 1])) & nan_condition))
        ] = (-j - 1)
        k = j

    values_out[np.where(((median_value + width[k + 1]) < band) & nan_condition)] = k + 1
    values_out[np.where(((median_value - width[k + 1]) > band) & nan_condition)] = -k - 1
    values_out[np.where(median_value == band)] = 0

    out_image.append(values_out)

    return out_image


@beartype
def raster_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, number_of_classes: int, nan_value: Union[int, float], bands: Sequence[int]
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with geometrical intervals (Torppa, 2023).

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_classes: The number of classes. The true number of classes is at most double the amount,
        depending how symmetrical the input data is.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with geometrical intervals (Torppa, 2023) and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        geometrical_intervals_band = _raster_with_geometrical_intervals(band, number_of_classes, nan_value)

        out_image.append(geometrical_intervals_band)

    return out_image, out_meta


def _raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    band: np.ndarray,
    number_of_intervals: int,
) -> np.ndarray:

    out_image = []

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

    out_image.append(classified)

    return out_image


@beartype
def raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Sequence[int],
) -> Tuple[Sequence[np.ndarray], dict]:
    """Classify raster with standard deviation.

    If bands are not given, all bands are used for classification.

    Args:
        raster: Raster to be classified.
        number_of_intervals: The number of intervals.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        Raster classified with standard deviation and metadata.

    Raises:
        InvalidParameterValueException: Bands contain negative values.
    """

    out_image = []
    out_meta = raster.meta.copy()

    bands_to_read = bands if bands is not None else raster.indexes

    _bands_non_negative(bands_to_read)

    for band in raster.read(bands_to_read):

        standard_deviation_band = _raster_with_standard_deviation(band, number_of_intervals)

        out_image.append(standard_deviation_band)

    return out_image, out_meta
