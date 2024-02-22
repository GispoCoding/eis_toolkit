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
    cast_array_to_int,
    cast_scalar_to_int,
    expand_and_zip,
)
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _clip_transform(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
    limits: Tuple[Optional[Number], Optional[Number]],
) -> np.ndarray:
    limit_lower, limit_upper = limits[0], limits[1]

    out_array = in_array

    if limit_lower is not None:
        out_array = np.where(out_array < limit_lower, limit_lower, out_array)

    if limit_upper is not None:
        out_array = np.where(out_array > limit_upper, limit_upper, out_array)

    return out_array


@beartype
def clip_transform(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    limits: Sequence[Tuple[Optional[Number], Optional[Number]]],
    bands: Optional[Sequence[int]] = None,
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Clips data based on specified upper and lower limits.

    Takes one nodata value that will be ignored in calculations.
    Replaces values below the lower limit and above the upper limit with provided values, respecively.
    Works both one-sided and two-sided but raises error if no limits provided.

    If no band/column selection specified, all bands/columns will be used.
    If a parameter contains only 1 entry, it will be applied for all bands.
    The limits can be set for each band individually.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        limits: Lower and upper limits (lower, upper) as real values.
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

    if check_parameter_length(bands, limits) is False:
        raise NonMatchingParameterLengthsException("Invalid limit length.")

    for item in limits:
        if item.count(None) == len(item):
            raise InvalidParameterValueException(f"Limit values all None: {item}.")

        if not check_minmax_position(item):
            raise InvalidParameterValueException(f"Invalid min-max values provided: {item}.")

    expanded_args = expand_and_zip(bands, limits)
    limits = [element[1] for element in expanded_args]

    out_settings = {}

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        inital_dtype = band_array.dtype

        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = nodata_to_nan(band_array, nodata_value=nodata)

        band_array = _clip_transform(band_array, limits=limits[i])

        band_array = nan_to_nodata(band_array, nodata_value=nodata)
        band_array = cast_array_to_int(band_array, scalar=nodata, initial_dtype=inital_dtype)

        band_array = np.expand_dims(band_array, axis=0)

        if i == 0:
            out_array = band_array.copy()
        else:
            out_array = np.vstack((out_array, band_array))

        current_transform = f"transformation {i + 1}"
        current_settings = {
            "band_origin": bands[i],
            "limit_lower": cast_scalar_to_int(limits[i][0]),
            "limit_upper": cast_scalar_to_int(limits[i][1]),
            "nodata": cast_scalar_to_int(nodata),
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
