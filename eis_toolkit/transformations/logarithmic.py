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
    expand_and_zip,
    replace_values,
    set_max_precision,
    truncate_decimal_places,
)
from eis_toolkit.utilities.nodata import nan_to_nodata


@beartype
def _log_transform_ln(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
) -> np.ndarray:

    return np.log(in_array)


@beartype
def _log_transform_log2(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
) -> np.ndarray:

    return np.log2(in_array)


@beartype
def _log_transform_log10(  # type: ignore[no-any-unimported]
    in_array: np.ndarray,
) -> np.ndarray:

    return np.log10(in_array)


@beartype
def log_transform(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    log_transform: Sequence[str] = ["log2"],
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Perform a logarithmic transformation on the provided data.

    Takes one nodata value that will be ignored in calculations.
    Negative values will not be considered for transformation and replaced by the specific nodata value.

    If no band/column selection specified, all bands/columns will be used.
    If a parameter contains only 1 entry, it will be applied for all bands.
    The log_transform can be set for each band individually.

    Args:
        raster: Data object to be transformed.
        bands: Selection of bands to be transformed.
        log_transform: The base for logarithmic transformation. Valid values 'ln', 'log2' and 'log10'.
        nodata: Nodata value to be considered.

    Returns:
        out_array: The transformed data.
        out_meta: Updated metadata.
        out_settings: Log of input settings and calculated statistics if available.

    Raises:
        InvalidRasterBandException: The input contains invalid band numbers.
        NonMatchingParameterLengthsException: The input does not match the number of selected bands
        InvalidParameterValueException: The input does not match the requirements (values, order of values)
    """
    bands = list(range(1, raster.count + 1)) if bands is None else bands
    nodata = raster.nodata if nodata is None else nodata

    if check_raster_bands(raster, bands) is False:
        raise InvalidRasterBandException("Invalid band selection")

    if check_parameter_length(bands, log_transform) is False:
        raise NonMatchingParameterLengthsException("Invalid length for log-base values.")

    for item in log_transform:
        if not (item == "ln" or item == "log2" or item == "log10"):
            raise InvalidParameterValueException(f"Invalid method: {item}.")

    expanded_args = expand_and_zip(bands, log_transform)
    log_transform = [element[1] for element in expanded_args]

    out_settings = {}
    out_decimals = set_max_precision()

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])
        band_array = cast_array_to_float(band_array, cast_int=True)
        band_array = replace_values(band_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)
        band_array[band_array <= 0] = np.nan

        if log_transform[i] == "ln":
            band_array = _log_transform_ln(band_array.astype(np.float64))
        elif log_transform[i] == "log2":
            band_array = _log_transform_log2(band_array.astype(np.float64))
        elif log_transform[i] == "log10":
            band_array = _log_transform_log10(band_array.astype(np.float64))

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
            "log_transform": log_transform[i],
            "nodata": nodata,
            "decimal_places": out_decimals,
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
