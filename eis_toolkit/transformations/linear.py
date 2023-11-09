from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.utilities.checks.parameter import check_minmax_position, check_parameter_length
from eis_toolkit.utilities.checks.raster import check_raster_bands
from eis_toolkit.utilities.miscellaneous import (
    cast_array_to_float,
    expand_and_zip,
    replace_values,
    set_max_precision,
    truncate_decimal_places,
)
from eis_toolkit.utilities.nodata import nan_to_nodata


@beartype
def _z_score_normalization(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
) -> Tuple[np.ndarray, Number, Number]:
    mean = np.nanmean(in_array)
    sd = np.nanstd(in_array)

    out_array = (in_array - mean) / sd

    return out_array, mean, sd


@beartype
def _min_max_scaling(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
    new_range: Tuple[Number, Number],
) -> np.ndarray:
    array_min = np.nanmin(in_array)
    array_max = np.nanmax(in_array)
    scaled_min, scaled_max = new_range[0], new_range[1]

    scaler = (in_array - array_min) / (array_max - array_min)
    out_array = (scaler * (scaled_max - scaled_min)) + scaled_min

    return out_array


@beartype
def z_score_normalization(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Normalize data based on mean and standard deviation.

    Results will have a mean = 0 and standard deviation = 1.
    Takes one nodata value that will be ignored in calculations.

    If no band/column selection specified, all bands/columns will be used.
    If a parameter contains only 1 entry, it will be applied for all bands.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        nodata: Nodata value to be considered.

    Returns:
        out_array: The transformed data.
        out_meta: Updated metadata.
        out_settings: Log of input settings and calculated statistics if available.

    Raises:
        InvalidRasterBandException: The input contains invalid band numbers.
        NonMatchingParameterLengthsException: The input does not match the number of selected bands.
    """
    bands = list(range(1, raster.count + 1)) if bands is None else bands
    nodata = raster.nodata if nodata is None else nodata

    if check_raster_bands(raster, bands) is False:
        raise InvalidRasterBandException("Invalid band selection.")

    out_settings = {}
    out_decimals = set_max_precision()

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = replace_values(band_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)

        band_array, mean_array, sd_array = _z_score_normalization(band_array.astype(np.float64))

        band_array = truncate_decimal_places(band_array, decimal_places=out_decimals)
        band_array = nan_to_nodata(band_array, nodata_value=nodata)
        band_array = cast_array_to_float(band_array, scalar=nodata, cast_float=True)

        band_array = np.expand_dims(band_array, axis=0)

        if i == 0:
            out_array = band_array.copy()
        else:
            out_array = np.vstack((out_array, band_array))

        current_transform = f"transformation {i + 1}"
        current_settings = {
            "band_origin": bands[i],
            "original_mean": truncate_decimal_places(mean_array, decimal_places=out_decimals),
            "original_sd": truncate_decimal_places(sd_array, decimal_places=out_decimals),
            "nodata": nodata,
            "decimal_places": out_decimals,
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings


@beartype
def min_max_scaling(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    new_range: Sequence[Tuple[Number, Number]] = [(0, 1)],
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Normalize data based on a specified new range.

    Uses the provided new minimum and maximum to transform data into the new interval.
    Takes one nodata value that will be ignored in calculations.

    If no band/column selection specified, all bands/columns will be used.
    The new_range can be set for each band individually.
    If a parameter contains only 1 entry, it will be applied for all bands.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        new_range: The new interval data will be transformed into. First value corresponds to min, second to max.
        nodata: Nodata value to be considered.

    Returns:
        out_array: The transformed data.
        out_meta: Updated metadata.
        out_settings: Log of input settings and calculated statistics if available.

    Raises:
        InvalidRasterBandException: The input contains invalid band numbers.
        NonMatchingParameterLengthsException: The input does not match the number of selected bands.
        InvalidParameterValueException: The input does not match the requirements (values, order of values).
    """
    bands = list(range(1, raster.count + 1)) if bands is None else bands
    nodata = raster.nodata if nodata is None else nodata

    if check_raster_bands(raster, bands) is False:
        raise InvalidRasterBandException("Invalid band selection")

    if check_parameter_length(bands, new_range) is False:
        raise NonMatchingParameterLengthsException("Invalid new_range length")

    for item in new_range:
        if not check_minmax_position(item):
            raise InvalidParameterValueException(f"Invalid min-max values provided: {item}")

    expanded_args = expand_and_zip(bands, new_range)
    new_range = [element[1] for element in expanded_args]

    out_settings = {}
    out_decimals = set_max_precision()

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = replace_values(band_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)

        band_array = _min_max_scaling(band_array.astype(np.float64), new_range=new_range[i])

        band_array = truncate_decimal_places(band_array, decimal_places=out_decimals)
        band_array = nan_to_nodata(band_array, nodata_value=nodata)
        band_array = cast_array_to_float(band_array, scalar=nodata, cast_float=True)

        band_array = np.expand_dims(band_array, axis=0)

        if i == 0:
            out_array = band_array.copy()
        else:
            out_array = np.vstack((out_array, band_array))

        current_transform = f"transformation {i + 1}"
        current_settings = {
            "band_origin": bands[i],
            "scaled_min": new_range[i][0],
            "scaled_max": new_range[i][1],
            "nodata": nodata,
            "decimal_places": out_decimals,
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
