import os
from numbers import Number

import geopandas
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Tuple
from rasterio.io import MemoryFile
from scipy.ndimage import binary_dilation

from eis_toolkit.exceptions import EmptyDataFrameException, NonMatchingCrsException
from eis_toolkit.raster_processing.create_constant_raster import create_constant_raster
from eis_toolkit.utilities.checks.raster import check_matching_crs


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
    buffer_value: Optional[str] = None,
    overwrite_nodata: bool = False,
) -> np.ndarray:
    out_array = np.copy(array)
    out_array = _create_local_buffer(
        array=out_array,
        radius=radius,
        target_value=target_value,
    )

    if buffer_value == "avg":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where((array != 0) & (out_array != 0), (array + out_array) * 0.5, (array + out_array))
    elif buffer_value == "max":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where(array != 0, np.maximum(array, out_array), out_array)
    elif buffer_value == "min":
        out_array = np.where(out_array, target_value, 0)
        out_array = np.where((array != 0) & (out_array != 0), np.minimum(array, out_array), (array + out_array))
    else:
        out_array = np.where(out_array, target_value, array)

    if overwrite_nodata is False:
        out_array = np.where(np.isnan(array), np.nan, out_array)

    return out_array


@beartype
def save_raster(path: str, array: np.ndarray, meta: dict = None, overwrite: bool = False):
    """Save the given raster NumPy array and metadata in a raster format at the provided path.

    Args:
        path: Path to store the raster.
        array: Raster NumPy array.
        meta: Raster metadata.
        overwrite: overwrites the existing raster file if present, default false.
    """

    if os.path.exists(path) and overwrite is False:
        print(f"File already exists: {os.path.basename(path)}.")
        return
    else:
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)

    with rasterio.open(path, "w", compress="lzw", **meta) as dst:
        dst.write(array)
        dst.close()


def _point_to_raster(raster_array, raster_meta, positives, attribute, radius, buffer_value):
    with MemoryFile() as memfile:
        raster_meta["driver"] = "GTiff"
        with memfile.open(**raster_meta) as datawriter:
            datawriter.write(raster_array, 1)

        with memfile.open() as memraster:
            if not check_matching_crs(
                objects=[memraster, positives],
            ):
                raise NonMatchingCrsException("The raster and geodataframe are not in the same CRS.")

            # Select only positives that are within raster bounds
            positives = positives.cx[
                memraster.bounds.left : memraster.bounds.right,  # noqa: E203
                memraster.bounds.bottom : memraster.bounds.top,  # noqa: E203
            ]

            if attribute is not None:
                values = positives[attribute]
            else:
                values = [1]

            positives_rows, positives_cols = rasterio.transform.rowcol(
                memraster.transform, positives.geometry.x, positives.geometry.y
            )
            raster_array[positives_rows, positives_cols] = values

            unique_values = list(set(values))

            if radius is not None:
                for target_value in unique_values:
                    raster_array = _create_buffer_around_labels(raster_array, radius, target_value, buffer_value)

    return raster_array, raster_meta


@beartype
def points_to_raster(
    positives: geopandas.GeoDataFrame,
    attribute: Optional[str] = None,
    radius: Optional[int] = None,
    template_raster: Optional[rasterio.io.DatasetReader] = None,
    coord_west: Optional[Number] = None,
    coord_north: Optional[Number] = None,
    coord_east: Optional[Number] = None,
    coord_south: Optional[Number] = None,
    target_epsg: Optional[int] = None,
    target_pixel_size: Optional[int] = None,
    raster_width: Optional[int] = None,
    raster_height: Optional[int] = None,
    nodata_value: Optional[Number] = None,
    buffer_value: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """Convert a point data set into a binary raster.

    Assigning a value of 1 to pixels corresponding to the points and 0 elsewhere.
    Provide 3 methods for raster creation:
    1. Set extent and coordinate system based on a template raster.
    2. Set extent from origin, based on the western and northern coordinates and the pixel size.
    3. Set extent from bounds, based on western, northern, eastern and southern points.

    Always provide values for height and width for the last two options, which correspond to
    the desired number of pixels for rows and columns.

    Args:
        positives: The geodataframe points set to be converted into raster.
        template_raster: An optional raster to use as a template for the output.
        coord_west: The western coordinate of the output raster in [m].
        coord_east: The eastern coordinate of the output raster in [m].
        coord_south: The southern coordinate of the output raster in [m].
        coord_north: The northern coordinate of the output raster in [m].
        target_epsg: The EPSG code for the output raster.
        target_pixel_size: The pixel size of the output raster.
        raster_width: The width of the output raster.
        raster_height: The height of the output raster.
        nodata_value: The nodata value of the output raster.

    Returns:
        A tuple containing the output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataFrameException:  The input GeoDataFrame is empty.
        NonMatchingCrsException: The raster and geodataframe are not in the same CRS.
    """

    if positives.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    base_value = 0
    raster_array, raster_meta = create_constant_raster(
        base_value,
        template_raster,
        coord_west,
        coord_north,
        coord_east,
        coord_south,
        target_epsg,
        target_pixel_size,
        raster_width,
        raster_height,
        nodata_value,
    )

    raster_array, raster_meta = _point_to_raster(raster_array, raster_meta, positives, attribute, radius, buffer_value)

    return raster_array, raster_meta
