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
from eis_toolkit.utilities.checks.parameter import check_parameter_length
from eis_toolkit.utilities.checks.raster import check_raster_bands
from eis_toolkit.utilities.miscellaneous import (
    cast_array_to_float,
    cast_array_to_int,
    cast_scalar_to_int,
    expand_and_zip,
)
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _winsorize(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
    percentiles: Tuple[Optional[Number], Optional[Number]],
    inside: bool,
) -> Tuple[np.ndarray, Optional[Number], Optional[Number]]:
    percentile_lower, percentile_upper = percentiles[0], percentiles[1]
    calculated_lower, calculated_upper = None, None

    if inside is True:
        method_lower = "lower"
        method_upper = "higher"
    elif inside is False:
        method_lower = "higher"
        method_upper = "lower"

    out_array = in_array
    clean_array = np.extract(np.isfinite(in_array), in_array)

    if percentile_lower is not None:
        calculated_lower = np.percentile(clean_array, percentile_lower, method=method_lower)
        out_array = np.where(out_array < calculated_lower, calculated_lower, out_array)

    if percentile_upper is not None:
        calculated_upper = np.percentile(clean_array, 100 - percentile_upper, method=method_upper)
        out_array = np.where(out_array > calculated_upper, calculated_upper, out_array)

    return out_array, calculated_lower, calculated_upper


@beartype
def winsorize(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    percentiles: Sequence[Tuple[Optional[Number], Optional[Number]]],
    bands: Optional[Sequence[int]] = None,
    inside: bool = False,
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Winsorize data based on specified percentile values.

    Takes one nodata value that will be ignored in calculations.
    Replaces values between [minimum, lower percentile] and [upper percentile, maximum] if provided.
    Works both one-sided and two-sided but raises error if no percentile values provided.

    Percentiles are symmetrical, i.e. percentile_lower = 10 corresponds to the interval [min, 10%].
    And percentile_upper = 10 corresponds to the intervall [90%, max].
    I.e. percentile_lower = 0 refers to the minimum and percentile_upper = 0 to the data maximum.

    Calculation of percentiles is ambiguous. Users can choose whether to use the value
    for replacement from inside or outside of the respective interval. Example:
    Given the np.array[5 10 12 15 20 24 27 30 35] and percentiles(10, 10), the calculated
    percentiles are (5, 35) for inside and (10, 30) for outside.
    This results in [5 10 12 15 20 24 27 30 35] and [10 10 12 15 20 24 27 30 30], respectively.

    If no band/column selection specified, all bands/columns will be used.
    If a parameter contains only 1 entry, it will be applied for all bands.
    The percentiles can be set for each band individually, but inside parameter is same for all bands.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        percentiles: Lower and upper percentile values (lower, upper) between [0, 100].
        inside: Whether to use the value for replacement from the left or right of the calculated percentile.
        nodata: Nodata value to be considered.

    Returns:
        out_array: The transformed data.
        out_meta: Updated metadata.
        out_settings: Log of input settings and calculated statistics if available.

    Raises:
        InvalidRasterBandException: The input contains invalid band numbers.
        NonMatchingParameterLengthsException: The input does not match the number of selected bands.
        InvalidParameterValueException: The input does not match the requirements (values, order of values)
    """
    bands = list(range(1, raster.count + 1)) if bands is None else bands
    nodata = raster.nodata if nodata is None else nodata

    if check_raster_bands(raster, bands) is False:
        raise InvalidRasterBandException("Invalid band selection")

    if check_parameter_length(bands, percentiles) is False:
        raise NonMatchingParameterLengthsException("Invalid length for percentiles.")

    for item in percentiles:
        if item.count(None) == len(item):
            raise InvalidParameterValueException(f"Percentile values all None: {item}.")

        if None not in item and sum(item) >= 100:
            raise InvalidParameterValueException(f"Sum >= 100: {item}.")

        if item[0] is not None and not (0 < item[0] < 100):
            raise InvalidParameterValueException(f"Invalid lower percentile value: {item}.")

        if item[1] is not None and not (0 < item[1] < 100):
            raise InvalidParameterValueException(f"Invalid upper percentile value: {item}.")

    expanded_args = expand_and_zip(bands, percentiles)
    percentiles = [element[1] for element in expanded_args]

    out_settings = {}

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        inital_dtype = band_array.dtype

        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = nodata_to_nan(band_array, nodata_value=nodata)

        band_array, calculated_lower, calculated_upper = _winsorize(
            band_array, percentiles=percentiles[i], inside=inside
        )

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
            "percentile_lower": cast_scalar_to_int(percentiles[i][0]),
            "percentile_upper": cast_scalar_to_int(percentiles[i][1]),
            "calculated_lower": cast_scalar_to_int(calculated_lower),
            "calculated_upper": cast_scalar_to_int(calculated_upper),
            "nodata": cast_scalar_to_int(nodata),
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
