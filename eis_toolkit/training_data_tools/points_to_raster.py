from numbers import Number

import geopandas
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from rasterio import profiles, transform
from scipy.ndimage import binary_dilation

from eis_toolkit.exceptions import EmptyDataFrameException, NonMatchingCrsException
from eis_toolkit.utilities.checks.raster import check_raster_profile


def _get_kernel_size(radius: int) -> tuple[int, int]:
    size = 1 + (radius * 2)
    return size, radius


def _create_grid(size: int, radius) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(-radius, size - radius)
    x = np.arange(-radius, size - radius)
    y, x = np.meshgrid(y, x)
    return x, y


def _basic_kernel(radius: int, value: Number) -> np.ndarray:
    size, _ = _get_kernel_size(radius)

    x, y = _create_grid(size, radius)
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size))
    kernel[mask] = value

    return kernel


def _create_local_buffer(
    array: np.ndarray,
    radius: int,
    target_value: Number,
) -> np.ndarray:
    kernel = _basic_kernel(radius, target_value)
    array = np.squeeze(array) if array.ndim >= 3 else array

    return binary_dilation(array == target_value, structure=kernel)


def _create_buffer_around_labels(
    array: np.ndarray,
    radius: int = 1,
    target_value: int = 1,
    buffer: Optional[str] = None,
    overwrite_nodata: bool = False,
) -> np.ndarray:
    out_array = np.copy(array)
    out_array = _create_local_buffer(
        array=out_array,
        radius=radius,
        target_value=target_value,
    )

    if buffer == "avg":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where((array != 0) & (out_array != 0), (array + out_array) * 0.5, (array + out_array))
    elif buffer == "max":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where(array != 0, np.maximum(array, out_array), out_array)
    elif buffer == "min":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where((array != 0) & (out_array != 0), np.minimum(array, out_array), (array + out_array))
    else:
        out_array = np.where(out_array, target_value, array)

    if overwrite_nodata is False:
        out_array = np.where(np.isnan(array), np.nan, out_array)

    return out_array


def _point_to_raster(raster_array, raster_meta, geodataframe, attribute, radius, buffer):

    width = raster_meta.get("width")
    height = raster_meta.get("height")

    raster_transform = raster_meta.get("transform")

    left = raster_transform[2]
    top = raster_transform[5]
    right = left + width * raster_transform[0]
    bottom = top + height * raster_transform[4]

    geodataframe = geodataframe.cx[left:right, bottom:top]

    if attribute is not None:
        values = geodataframe[attribute]
    else:
        values = [1]

    positives_rows, positives_cols = transform.rowcol(
        raster_transform, geodataframe.geometry.x, geodataframe.geometry.y
    )
    raster_array[positives_rows, positives_cols] = values

    unique_values = list(set(values))

    if radius is not None:
        for target_value in unique_values:
            raster_array = _create_buffer_around_labels(raster_array, radius, target_value, buffer)

    return raster_array


@beartype
def points_to_raster(
    geodataframe: geopandas.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    attribute: Optional[str] = None,
    radius: Optional[int] = None,
    buffer: Optional[Literal["min", "avg", "max"]] = None,
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Convert a point data set into a binary raster.

    Assigs attribute values if provided else 1 to pixels corresponding to the points and 0 elsewhere.

    Args:
        geodataframe: The geodataframe points set to be converted into raster.
        attribute: Values to be be assigned to the geodataframe.
        radius: Radius to be applied around the geodataframe in [m].
        buffer: Buffers the matrix value when two or more radii with different attribute value overlap.
                'avg': performs averaging of the two attribute value
                'min': minimum of the two attribute values
                'max': maximum of the two attribute values

    Returns:
        A tuple containing the output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataFrameException:  The input GeoDataFrame is empty.
        NonMatchingCrsException: The raster and geodataframe are not in the same CRS.
    """

    if geodataframe.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")

    check_raster_profile(raster_profile=raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")

    raster_array = np.zeros((raster_height, raster_width))

    out_array = _point_to_raster(raster_array, raster_profile, geodataframe, attribute, radius, buffer)

    return out_array, raster_profile
