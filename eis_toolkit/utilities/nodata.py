import functools
from numbers import Number
from typing import Any, Union

import numpy as np
from beartype import beartype
from beartype.typing import Callable, Dict, Sequence

from eis_toolkit.exceptions import InvalidRasterBandException
from eis_toolkit.utilities.miscellaneous import replace_values


@beartype
def set_nodata_raster_meta(raster_meta: Dict, nodata_value: Number) -> Dict:
    """
    Set new nodata value for raster metadata.

    Note that this function does not convert any data values, only changes/fixes metadata.

    Args:
        raster_meta: Raster metadata to be updated.
        nodata_value: Nodata value to be set.

    Returns:
        Raster metadata with updated nodata value.
    """
    out_meta = raster_meta.copy()
    out_meta.update({"nodata": nodata_value})
    return out_meta


@beartype
def replace_raster_nodata_each_band(
    raster_data: np.ndarray,
    nodata_per_band: Dict[int, Union[Number, Sequence[Number]]],
    new_nodata: Number = -9999,  # type: ignore
) -> np.ndarray:
    """
    Replace old nodata values with a new nodata value in a raster for each band separately.

    Args:
        raster_data: Multiband raster's data.
        nodata_per_band: Mapping of bands and their current nodata values.
        new_nodata: A new nodata value that will be used for all old nodata values and all bands. Defaults to -9999.

    Returns:
        The original raster data with replaced nodata values.

    Raises:
        InvalidRasterBandException: Invalid band index in nodata mapping.
    """
    if any(band > len(raster_data) or band < 1 for band in nodata_per_band.keys()):
        raise InvalidRasterBandException("Invalid band index in nodata mapping.")

    out_raster_data = raster_data.copy()

    for band, nodata_values in nodata_per_band.items():
        index = band - 1
        band_data = raster_data[index]
        out_data = replace_values(band_data, nodata_values, new_nodata)
        out_raster_data[index] = out_data

    return out_raster_data


@beartype
def nodata_to_nan(data: np.ndarray, nodata_value: Union[Number, None]) -> np.ndarray:
    """Convert specified nodata_value to np.nan.

    Args:
        data: Input data as a numpy array.
        nodata_value: Value that is converted to np.nan.

    Returns:
        Input array where specified nodata has been converted to np.nan.
    """
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)

    return np.where(np.isin(data, nodata_value), np.nan, data)  # type: ignore


@beartype
def nan_to_nodata(data: np.ndarray, nodata_value: Number) -> np.ndarray:
    """Convert np.nan values to specified nodata_value.

    Args:
        data: Input data as a numpy array.
        nodata_value: Value that np.nan is converted to.

    Returns:
        Input array where np.nan has been converted to specified nodata.
    """
    return np.where(np.isnan(data), nodata_value, data)  # type: ignore


@beartype
def handle_nodata_as_nan(func: Callable) -> Callable:
    """Replace nodata_values with np.nan for function execution and reverses the replacement afterwards."""

    @functools.wraps(func)
    def wrapper(in_data: np.ndarray, *args: Any, nodata_value: Number, **kwargs: Any) -> np.ndarray:
        replaced_data = nodata_to_nan(in_data, nodata_value)
        result = func(replaced_data, *args, **kwargs)
        out_data = nan_to_nodata(result, nodata_value)
        return out_data

    return wrapper
