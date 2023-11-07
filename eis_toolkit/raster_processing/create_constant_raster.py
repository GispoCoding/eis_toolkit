from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Tuple
from rasterio.transform import from_bounds, from_origin

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.utilities.checks.parameter import check_minmax_position
from eis_toolkit.utilities.miscellaneous import cast_scalar_to_int, get_min_int_type


@beartype
def _create_constant_raster_from_template(
    constant_value: Number,
    template_raster: rasterio.io.DatasetReader,
    nodata_value: Optional[Number],
) -> Tuple[np.ndarray, dict]:

    out_meta = template_raster.meta.copy()
    del out_meta["driver"]
    if nodata_value is not None:
        out_meta["nodata"] = nodata_value

    out_array = np.full((out_meta["height"], out_meta["width"]), constant_value)

    return out_array, out_meta


@beartype
def _create_constant_raster_from_origin(
    constant_value: Number,
    coord_west: Number,
    coord_north: Number,
    target_epsg: int,
    target_pixel_size: int,
    raster_width: int,
    raster_height: int,
    nodata_value: Number,
) -> Tuple[np.ndarray, dict]:

    out_array = np.full((raster_height, raster_width), constant_value)
    out_transform = from_origin(coord_west, coord_north, target_pixel_size, target_pixel_size)
    out_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    out_meta = {
        "width": raster_width,
        "height": raster_height,
        "count": 1,
        "dtype": out_array.dtype,
        "crs": out_crs,
        "transform": out_transform,
        "nodata": nodata_value,
    }

    return out_array, out_meta


@beartype
def _create_constant_raster_from_bounds(
    constant_value: Number,
    coord_west: Number,
    coord_north: Number,
    coord_east: Number,
    coord_south: Number,
    target_epsg: int,
    raster_width: int,
    raster_height: int,
    nodata_value: Number,
) -> Tuple[np.ndarray, dict]:

    out_array = np.full((raster_height, raster_width), constant_value)
    out_transform = from_bounds(coord_west, coord_south, coord_east, coord_north, raster_width, raster_height)
    out_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    out_meta = {
        "width": raster_width,
        "height": raster_height,
        "count": 1,
        "dtype": out_array.dtype,
        "crs": out_crs,
        "transform": out_transform,
        "nodata": nodata_value,
    }

    return out_array, out_meta


@beartype
def create_constant_raster(  # type: ignore[no-any-unimported]
    constant_value: Number,
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
) -> Tuple[np.ndarray, dict]:
    """Create a constant raster based on a user-defined value.

    Provide 3 methods for raster creation:
    1. Set extent and coordinate system based on a template raster.
    2. Set extent from origin, based on the western and northern coordinates and the pixel size.
    3. Set extent from bounds, based on western, northern, eastern and southern points.

    Always provide values for height and width for the last two options, which correspond to
    the desired number of pixels for rows and columns.

    Args:
        constant_value: The constant value to use in the raster.
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
        InvalidParameterValueException: Provide invalid input parameter.
    """

    if template_raster is not None:
        out_array, out_meta = _create_constant_raster_from_template(constant_value, template_raster, nodata_value)

    elif all(coords is not None for coords in [coord_west, coord_east, coord_south, coord_north]):
        if raster_height <= 0 or raster_width <= 0:
            raise InvalidParameterValueException("Invalid raster extent provided.")
        if not check_minmax_position((coord_west, coord_east) or not check_minmax_position((coord_south, coord_north))):
            raise InvalidParameterValueException("Invalid coordinate values provided.")

        out_array, out_meta = _create_constant_raster_from_bounds(
            constant_value,
            coord_west,
            coord_north,
            coord_east,
            coord_south,
            target_epsg,
            raster_width,
            raster_height,
            nodata_value,
        )

    elif all(coords is not None for coords in [coord_west, coord_north]) and all(
        coords is None for coords in [coord_east, coord_south]
    ):
        if raster_height <= 0 or raster_width <= 0:
            raise InvalidParameterValueException("Invalid raster extent provided.")
        if target_pixel_size <= 0:
            raise InvalidParameterValueException("Invalid pixel size.")

        out_array, out_meta = _create_constant_raster_from_origin(
            constant_value,
            coord_west,
            coord_north,
            target_epsg,
            target_pixel_size,
            raster_width,
            raster_height,
            nodata_value,
        )

    else:
        raise InvalidParameterValueException("Suitable parameter values were not provided for any of the 3 methods.")

    constant_value = cast_scalar_to_int(constant_value)
    nodata_value = cast_scalar_to_int(out_meta["nodata"])

    if isinstance(constant_value, int) and isinstance(nodata_value, int):
        target_dtype = np.result_type(get_min_int_type(constant_value), get_min_int_type(nodata_value))
        out_array = out_array.astype(target_dtype)
        out_meta["dtype"] = out_array.dtype
    elif isinstance(constant_value, int) and isinstance(nodata_value, float):
        out_array = out_array.astype(get_min_int_type(constant_value))
        out_meta["dtype"] = np.float64.__name__
    elif isinstance(constant_value, float):
        out_array = out_array.astype(np.float64)
        out_meta["dtype"] = out_array.dtype

    return out_array, out_meta
