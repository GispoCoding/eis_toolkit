import mapclassify as mc
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple

from eis_toolkit.exceptions import InvalidParameterValueException


def _raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    breaks: Sequence[int],
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []

    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(1, len(array_of_bands) + 1, 1).tolist()

    for i in range(len(bands)):

        data_array = array_of_bands[i]

        data = np.digitize(data_array, breaks)

        out_image.append(data)

    return out_image, out_meta


@beartype
def raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, breaks: Sequence[int], bands: Optional[Sequence[int]]
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster, breaks is not a list of integers.
    """
    if bands is not None:
        if not isinstance(bands, Sequence):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")
    if not all(isinstance(_break, int) for _break in breaks):
        raise InvalidParameterValueException("Expected breaks to contain only integers")

    out_image, out_meta = _raster_with_manual_breaks(raster, breaks, bands)

    return out_image, out_meta


def _raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    interval_size: int,
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []
    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(1, len(array_of_bands) + 1, 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]

        _, edges = np.histogram(data_array, bins=interval_size)

        data = np.digitize(data_array, edges)

        out_image.append(data)

    return out_image, out_meta


@beartype
def raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, interval_size: int, bands: Optional[Sequence[int]]
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")

    out_image, out_meta = _raster_with_defined_intervals(raster, interval_size, bands)

    return out_image, out_meta


def _raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []
    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(1, len(array_of_bands) + 1, 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        percentiles = np.linspace(0, 100, number_of_intervals)
        intervals = np.percentile(data_array, percentiles)
        data = np.digitize(data_array, intervals)
        out_image.append(data)

    return out_image, out_meta


@beartype
def raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]],
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")

    out_image, out_meta = _raster_with_equal_intervals(raster, number_of_intervals, bands)

    return out_image, out_meta


def _raster_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []
    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(1, len(array_of_bands) + 1, 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        intervals = [np.percentile(data_array, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
        data = np.digitize(data_array, intervals)

        out_image.append(data)

    return out_image, out_meta


@beartype
def raster_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Optional[Sequence[int]],
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")

    out_image, out_meta = _raster_with_quantiles(raster, number_of_quantiles, bands)

    return out_image, out_meta


def _raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []
    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(1, len(array_of_bands) + 1, 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        breaks = mc.JenksCaspall(data_array, number_of_classes)
        data = np.digitize(data_array, np.sort(breaks.bins))

        out_image.append(data)

    return out_image, out_meta


@beartype
def raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, number_of_classes: int, bands: Optional[Sequence[int]]
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")

    out_image, out_meta = _raster_with_natural_breaks(raster, number_of_classes, bands)

    return out_image, out_meta


def _raster_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, number_of_classes: int, bands: Optional[Sequence[int]]
) -> Tuple[Sequence[np.ndarray], dict]:

    array_of_bands = []
    out_image = []
    out_meta = raster.meta.copy()

    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    for i in range(len(array_of_bands)):

        data_array = array_of_bands[i]
        # Masking missing values with np.nan
        data_array[data_array == -1.0e32] = np.nan

        median_value = np.nanmedian(data_array)
        max_value = np.nanmax(data_array)
        min_value = np.nanmin(data_array)

        data_array = np.ma.masked_where(np.isnan(data_array), data_array)
        values_out = np.zeros_like(data_array)  # The same shape as the original raster value array

        # Determine the tail with larger length
        if (median_value - min_value) < (max_value - median_value):  # Large end tail longer
            tail_values = data_array[np.where((data_array > median_value) & (data_array != np.nan))]
            range_tail = max_value - median_value
            tail_values = tail_values - median_value + range_tail / 1000.0
        else:  # Small end tail longer
            tail_values = data_array[np.where(data_array < median_value) & (data_array != np.nan)]
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
                np.where(
                    ((median_value + width[j]) < data_array)
                    & (data_array <= (median_value + width[j + 1]))
                    & (data_array != np.nan)
                )
            ] = (j + 1)
            values_out[
                np.where(
                    ((median_value - width[j]) > data_array)
                    & (data_array >= (median_value - width[j + 1]))
                    & (data_array != np.nan)
                )
            ] = (-j - 1)
            k = j

        values_out[np.where(((median_value + width[k + 1]) < data_array) & (data_array != np.nan))] = k + 1
        values_out[np.where(((median_value - width[k + 1]) > data_array) & (data_array != np.nan))] = -k - 1
        values_out[np.where(median_value == data_array)] = 0

        out_image.append(values_out)

    return out_image, out_meta


@beartype
def raster_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, number_of_classes: int, bands: Optional[Sequence[int]]
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")
    if number_of_classes <= 0:
        raise InvalidParameterValueException("The number of classes must be at least one")

    out_image, out_meta = _raster_with_geometrical_intervals(raster, number_of_classes, bands)

    return out_image, out_meta


def _raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]],
) -> Tuple[Sequence[np.ndarray], dict]:

    out_image = []
    out_meta = raster.meta.copy()

    band_statistics = []
    if bands is not None:
        for band in bands:
            stddev = np.nanstd(band)
            mean = np.nanmean(band)
            band_statistics.append((mean, stddev))
    else:
        for band in range(1, raster.count + 1):
            stddev = np.nanstd(band)
            mean = np.nanmean(band)
            band_statistics.append((mean, stddev))

    for band, (mean, std) in enumerate(band_statistics):
        data_array = raster.read(band + 1)
        interval_size = 2 * std / number_of_intervals

        classified = np.empty_like(data_array)

        below_mean = data_array < (mean - std)
        above_mean = data_array > (mean + std)

        classified[below_mean] = -number_of_intervals
        classified[above_mean] = number_of_intervals

        in_between = ~below_mean & ~above_mean
        interval = ((data_array - (mean - std)) / interval_size).astype(int)
        classified[in_between] = interval[in_between] - number_of_intervals // 2

        out_image.append(classified)

    return out_image, out_meta


@beartype
def raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[Sequence[int]],
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
        InvalidParameterValueException: Bands is not a list, bands is not a list of integers,
        bands exceeds the number of bands in the raster.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException("Expected bands parameter to be a list")
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException("Expected bands to be a list of integers")
        elif len(bands) > raster.count:
            raise InvalidParameterValueException("The number of bands given exceeds the number of raster's bands")

    out_image, out_meta = _raster_with_standard_deviation(raster, number_of_intervals, bands)

    return out_image, out_meta
