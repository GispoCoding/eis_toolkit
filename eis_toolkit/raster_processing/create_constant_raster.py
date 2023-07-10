import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Tuple, Optional, Sequence
from rasterio.transform import from_origin, from_bounds
from numbers import Number

from eis_toolkit.checks.parameter import check_minmax_position


@beartype
def _create_constant_raster_from_template(
    constant_value: Number,
    template_raster: rasterio.io.DatasetReader,
) -> Tuple[np.ndarray, dict]:

    out_meta = template_raster.meta.copy()
    out_array = np.full((out_meta.height, out_meta.width), constant_value)

    return out_array, out_meta


@beartype
def _create_constant_raster_from_origin(
    constant_value: Number,
    x_min: Number,
    y_max: Number,
    target_epsg: int,
    target_pixel_size: int,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, dict]:

    out_array = np.full((target_height, target_width), constant_value)
    out_transform = from_origin(x_min, y_max, target_pixel_size, target_pixel_size)
    out_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    out_meta = {
        "width": target_width,
        "height": target_height,
        "count": 1,
        "dtype": out_array.dtype,
        "crs": out_crs,
        "transform": out_transform,
    }

    return out_array, out_meta


@beartype
def _create_constant_raster_from_bounds(
    constant_value: Number,
    x_min: Number,
    x_max: Number,
    y_min: Number,
    y_max: Number,
    target_epsg: int,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, dict]:

    out_array = np.full((target_height, target_width), constant_value)
    out_transform = from_bounds(x_min, y_min, x_max, y_max, target_width, target_height)
    out_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    out_meta = {
        "width": target_width,
        "height": target_height,
        "count": 1,
        "dtype": out_array.dtype,
        "crs": out_crs,
        "transform": out_transform,
    }

    return out_array, out_meta


@beartype
def create_constant_raster(
    constant_value: Number,
    template_raster: Optional[rasterio.io.DatasetReader] = None,
    x_min: Optional[Number] = None,
    x_max: Optional[Number] = None,
    y_min: Optional[Number] = None,
    y_max: Optional[Number] = None,
    target_epsg: Optional[int] = None,
    target_pixel_size: Optional[int] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
):
    if template_raster is not None:
        out_array, out_meta = _create_constant_raster_from_template(constant_value, template_raster)
    else:
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:

            out_array, out_meta = _create_constant_raster_from_bounds(
                constant_value, x_min, x_max, y_min, y_max, target_epsg, target_width, target_height
            )
        else:
            out_array, out_meta = _create_constant_raster_from_origin(
                constant_value, x_min, y_max, target_epsg, target_pixel_size, target_width, target_height
            )


# def _create_constant_raster_from_template(
#     constant_value: Number,
#     target_epsg: Optional[int],
#     target_width: Optional[int],
#     target_height: Optional[int],
#     target_pixel_size: Optional[int],
#     xy_north_west: Optional[Sequence[Number, Number]],
#     xy_south_east: Optional[Sequence[Number, Number]],
#     template_raster: Optional[rasterio.io.DatasetReader],
# ) -> Tuple[np.ndarray, dict]:
#     if template_raster is not None:
#         out_meta = template_raster.meta.copy()
#         out_array = np.full((out_meta.height, out_meta.width), constant_value)
#     else:
#       if xy_north_west is not None and xy_south_east is None:
#         out_array = np.full((target_height, target_width), constant_value)
#         out_transform = from_origin(xy_north_west[0], xy_north_west[1], target_pixel_size, target_pixel_size)
#       elif xy_north_west is not None and xy_south_east is not None:

#         out_transform = from_bounds(xy_north_west[0], xy_south_east[1], xy_south_east[0], xy_north_west[1], target_width, target_height)


#     out_crs = rasterio.crs.CRS.from_epsg(target_epsg)
#     out_meta = {
#         "width": target_width,
#         "height": target_height,
#         "count": 1,
#         "dtype": out_array.dtype,
#         "crs": out_crs,
#         "transform": out_transform,
#     }

#     return out_array, out_meta
