import math
import numpy as np
import geopandas
import pandas as pd
from numbers import Number
from beartype import beartype
from rasterio import profiles, transform
from scipy.ndimage import binary_dilation
from beartype.typing import Literal, Optional, Tuple, Union

from eis_toolkit.utilities.checks.raster import check_raster_profile
from eis_toolkit.exceptions import EmptyDataFrameException, NonMatchingCrsException, InvalidColumnException, NonNumericDataException

def _convert_radius(radius: int, x: Number, y: Number) -> int:
    raster_radius = math.sqrt(x**2 + y**2) #RADIUS OF A SINGLE PIXEL
    r = radius/raster_radius
    return math.ceil(r) if r - math.floor(r) >= 0.5 else math.floor(r)

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
    target_value: Number = 1,
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
        x = raster_transform[0]
        y = raster_transform[4]
        radius = _convert_radius(radius,x,y)
        for target_value in unique_values:
            raster_array = _create_buffer_around_labels(raster_array, radius, target_value, buffer)

    return raster_array


@beartype
def points_to_raster(
    geodataframe: geopandas.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    attribute: Optional[str] = None,
    radius: Optional[Number] = None,
    buffer: Optional[Literal["min", "avg", "max"]] = None,
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Convert a GeoDataFrame of points into a binary raster using a provided base raster profile.

    Accepts a base raster profile and a geodataframe with points to be converted to binary raster.
    By default, the points are assigned a value of 1, and all other areas are set to 0. If an 
    attribute is provided, the raster will take the corresponding values from the attribute column 
    in the GeoDataFrame instead of 1. The base raster profile defines the template for the raster's
    extent, resolution, and projection. Optionally, a radius can be applied around each point (with 
    units consistent with the raster profile) to expand the point's influence within the raster. In 
    the case of overlapping radii with different attribute values, a buffer can be used to resolve 
    the conflict by selecting the minimum, maximum, or average value from the overlapping pixels.
    
    Args:
        geodataframe: The geodataframe points set to be converted into raster.
        attribute: Values to be be assigned to the geodataframe.
        radius: Radius to be applied around the geodataframe with units consistent with raster profile.
        buffer: Buffers the matrix value when two or more radii with different attribute value overlap.
                'avg': performs averaging of the two attribute value
                'min': minimum of the two attribute values
                'max': maximum of the two attribute values

    Returns:
        A tuple containing the output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataFrameException:  The input GeoDataFrame is empty.
        NonMatchingCrsException: The raster and geodataframe are not in the same CRS.
        InvalidColumnException: The attribute column was not found in geodataframe.
        NonNumericDataException: Some numeric parameters have invalid values.
    """

    if geodataframe.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    
    if attribute is not None:

        if attribute not in geodataframe.columns:
            raise InvalidColumnException(f"Attribute '{attribute}' not found in the geodataframe")
        
        if not pd.to_numeric(geodataframe[attribute], errors='coerce').notna().all():
            raise NonNumericDataException(f"Values in the '{attribute}' column are non numeric type")


    check_raster_profile(raster_profile=raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")

    raster_array = np.zeros((raster_height, raster_width))

    out_array = _point_to_raster(raster_array, raster_profile, geodataframe, attribute, radius, buffer)

    return out_array, raster_profile
