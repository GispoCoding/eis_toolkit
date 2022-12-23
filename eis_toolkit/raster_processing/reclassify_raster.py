from typing import List, Optional

import mapclassify as mc
import numpy as np
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    breaks: List[int],
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):

        data_array = array_of_bands[i]

        data = np.digitize(data_array, breaks)

        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_manual_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    breaks: List[int],
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with manual breaks.

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        breaks (List[int]): List of break values for the classification.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with manual breaks.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException
    if breaks is None:
        raise InvalidParameterValueException
    else:
        if not all(isinstance(_break, int) for _break in breaks):
            raise InvalidParameterValueException

    src = _raster_with_manual_breaks(raster, breaks, bands)

    return src


def _raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    interval_size: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetWriter:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]

        hist, edges = np.histogram(data_array, bins=interval_size)

        data = np.digitize(data_array, edges)

        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_defined_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    interval_size: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with defined intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        interval_size (int): The number of units in each interval.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with defined intervals.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_defined_intervals(raster, interval_size, bands)

    return src


def _raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        percentiles = np.linspace(0, 100, number_of_intervals)
        intervals = np.percentile(data_array, percentiles)
        data = np.digitize(data_array, intervals)
        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_equal_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with equal intervals.

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        number_of_intervals (int): The number of intervals.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with equal intervals.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_equal_intervals(raster, number_of_intervals, bands)

    return src


def _raster_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        intervals = [np.percentile(data_array, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
        data = np.digitize(data_array, intervals)

        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_quantiles(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_quantiles: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with quantiles.

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        number_of_quantiles: (int): The number of quantiles.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with quantiles.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_quantiles(raster, number_of_quantiles, bands)

    return src


def _raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):
        data_array = array_of_bands[i]
        breaks = mc.JenksCaspall(data_array, number_of_classes)
        data = np.digitize(data_array, np.sort(breaks.bins))

        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_natural_breaks(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with natural breaks (Jenks Caspall).

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        number_of_classes (int),: The number of classes.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with natural breaks (Jenks Caspall).
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_quantiles(raster, number_of_classes, bands)

    return src


def _raster_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):
        # read one of the bands
        data_array = array_of_bands[i]

        max_value = raster.statistics(i+1).max

        min_value = raster.statistics(i+1).min
        # get X according to https://www.mdpi.com/2673-4931/10/1/1/htm (Formula 2)
        x = (max_value/min_value)**(1/number_of_classes)
        # calculate intervals according to https://www.mdpi.com/2673-4931/10/1/1/htm (Formula 1)
        intervals = [min_value * x**j for j in range(1, number_of_classes)]

        # transform intervals that have become complex numbers to float
        intervals = [float(interval.real + interval.imag) for interval in intervals]

        # Classify the raster values into the intervals
        data = np.digitize(data_array, np.sort(intervals))

        # Write the data to the correct band
        if custom_band_list:
            raster.write(data, bands[i])
        else:
            raster.write(data, bands[i]+1)

    return raster


def raster_with_geometrical_intervals(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_classes: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with geometrical intervals (Francisci D., 2021).

    If bands are not given, all bands are used for classification.
    This algorithm is based on Francisci (2021) found here: https://doi.org/10.3390/environsciproc2021010001.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        number_of_classes (int),: The number of classes.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with geometrical intervals (Francisci D., 2021).
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_geometrical_intervals(raster, number_of_classes, bands)

    return src


def _raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:

    custom_band_list = False if bands is None else True
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()
        bands = np.arange(0, len(array_of_bands), 1).tolist()

    for i in range(len(bands)):

        data_array = array_of_bands[i]
        stddev = raster.statistics(i+1).std
        mean = raster.statistics(i+1).mean
        interval_size = 2 * stddev / number_of_intervals

        classified = np.empty_like(data_array)

        for j in range(data_array.shape[0]):
            for k in range(data_array.shape[1]):
                value = data_array[j, k]
                if value < mean - stddev:
                    classified[j, k] = -number_of_intervals
                elif value > mean + stddev:
                    classified[j, k] = number_of_intervals
                else:
                    interval = int((value - mean + stddev) / interval_size)
                    classified[j, k] = interval - number_of_intervals // 2

        if custom_band_list:
            raster.write(classified, bands[i])
        else:
            raster.write(classified, bands[i]+1)

    return raster


def raster_with_standard_deviation(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
    """Classify raster with standard deviation.

    If bands are not given, all bands are used for classification.

    Args:
        raster (rasterio.io.DatasetReader): Raster to be classified.
        number_of_intervals (int): The number of intervals.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.

    Returns:
        rasterio.io.DatasetReader: Raster classified with standard deviation.
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException
        elif len(bands) > raster.count:
            raise InvalidParameterValueException

    src = _raster_with_standard_deviation(raster, number_of_intervals, bands)

    return src
