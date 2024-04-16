import functools
from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from rasterio import profiles

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.utilities.miscellaneous import replace_values


@beartype
def unify_raster_nodata(
    input_rasters: Sequence[rasterio.io.DatasetReader], new_nodata: Number = -9999
) -> Sequence[Tuple[np.ndarray, dict]]:
    """
    Unifies nodata for the input rasters.

    All old nodata values in the input rasters are converted to the new nodata value. Raster metadatas
    is also updated with the new nodata value.

    Args:
        input_rasters:
        new_nodata: New nodata value that will be used to replace existing nodata for all bands in all
            input rasters. Defaults to -9999.

    Returns:
        Output raster list. List elements are tuples where first element is raster data and second \
        element is raster metadata.

    Raises:
        InvalidParameterValueException: Input raster list contains only one raster.
    """

    if len(input_rasters) < 2:
        raise InvalidParameterValueException(
            f"Expected multiple rasters in the input_rasters list. Rasters: {len(input_rasters)}. \
            To convert nodata of one raster, use the 'convert_raster_nodata' tool."
        )

    out_rasters = []
    for raster in input_rasters:
        out_image, out_meta = convert_raster_nodata(raster, new_nodata=new_nodata)
        out_rasters.append((out_image, out_meta))

    return out_rasters


@beartype
def set_raster_nodata(raster_meta: Union[Dict, profiles.Profile], new_nodata: Number) -> Union[Dict, profiles.Profile]:
    """
    Set new nodata value for raster metadata or profile.

    Note that this function does not convert any data values, it only changes the metadata.
    The inteded use case for this tool is fixing metadata with invalid nodata value.

    Args:
        raster_meta: Raster metadata or profile to be updated.
        nodata_value: New nodata value.

    Returns:
        Raster metadata / profile with updated nodata value.
    """
    out_meta = raster_meta.copy()
    out_meta.update({"nodata": new_nodata})
    return out_meta


@beartype
def convert_raster_nodata(
    input_raster: rasterio.io.DatasetReader,
    old_nodata: Optional[Number] = None,
    new_nodata: Number = -9999,
) -> Tuple[np.ndarray, dict]:
    """
    Convert existing nodata values with a new nodata value for a raster.

    Args:
        input_raster: Input raster dataset.
        new_nodata: New nodata value that will be used to replace existing nodata for all bands. Defaults to -9999.

    Returns:
        The input raster data and metadata updated with the new nodata.

    Raises:
        InvalidParameterValueException: Nodata is not defined in raster metadata and old_nodata was not specified.
    """
    if old_nodata is None and not input_raster.meta["nodata"]:
        raise InvalidParameterValueException(
            "Could not find old nodata value from raster metadata. Either define old_nodata or use \
            'set_raster_nodata' tool to fix broken raster metadata."
        )
    raster_arr = input_raster.read()
    old_nodata = input_raster.nodata if old_nodata is None else old_nodata
    out_image = replace_values(raster_arr, old_nodata, new_nodata)
    out_meta = input_raster.meta.copy()
    out_meta["nodata"] = new_nodata

    return out_image, out_meta


@beartype
def nodata_to_nan(data: np.ndarray, nodata_value: Number) -> np.ndarray:
    """Convert specified nodata_value to np.nan.

    Args:
        data: Input data as a numpy array.
        nodata_value: Value that is converted to np.nan.

    Returns:
        Input array where specified nodata has been converted to np.nan.
    """
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)

    return np.where(np.isin(data, nodata_value), np.nan, data)


@beartype
def nan_to_nodata(data: np.ndarray, nodata_value: Number) -> np.ndarray:
    """Convert np.nan values to specified nodata_value.

    Args:
        data: Input data as a numpy array.
        nodata_value: Value that np.nan is converted to.

    Returns:
        Input array where np.nan has been converted to specified nodata.
    """
    return np.where(np.isnan(data), nodata_value, data)


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
