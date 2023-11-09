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
def _sigmoid_transform(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
    bounds: Tuple[Number, Number],
    slope: Number,
    center: bool,
) -> np.ndarray:
    lower, upper = bounds[0], bounds[1]

    if center is True:
        in_array = in_array - np.nanmean(in_array)

    out_array = lower + (upper - lower) * (1 / (1 + np.exp(-slope * (in_array))))

    return out_array


@beartype
def sigmoid_transform(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    bounds: Sequence[Tuple[Number, Number]] = [(0, 1)],
    slope: Sequence[Number] = [1],
    center: bool = True,
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Transform data into a sigmoid-shape based on a specified new range.

    Uses the provided new minimum and maximum, shift and slope parameters to transform the data.
    Takes one nodata value that will be ignored in calculations.

    If no band/column selection specified, all bands/columns will be used.
    If a parameter contains only 1 entry, it will be applied for all bands.
    The bounds and slope values can be set for each band individually.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        bounds: Boundaries for the calculation of the sigmoid function (lower, upper).
        slope: Value which modifies the slope of the resulting sigmoid-curve.
        center: Center array values around mean = 0 before sigmoid transformation.
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

    for parameter_name, parameter in [("bounds", bounds), ("slope", slope)]:
        if check_parameter_length(bands, parameter) is False:
            raise NonMatchingParameterLengthsException(f"Invalid length for {parameter_name}.")

    for item in bounds:
        if check_minmax_position(item) is False:
            raise InvalidParameterValueException(f"Invalid min-max values provided: {item}.")

    expanded_args = expand_and_zip(bands, bounds, slope)
    bounds = [element[1] for element in expanded_args]
    slope = [element[2] for element in expanded_args]

    out_settings = {}
    out_decimals = set_max_precision()

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = replace_values(band_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)

        band_array = _sigmoid_transform(band_array.astype(np.float64), bounds=bounds[i], slope=slope[i], center=center)

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
            "bound_lower": truncate_decimal_places(bounds[i][0], decimal_places=out_decimals),
            "bound_upper": truncate_decimal_places(bounds[i][1], decimal_places=out_decimals),
            "slope": slope[i],
            "center": center,
            "nodata": nodata,
            "decimal_places": out_decimals,
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
