from math import ceil

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Tuple

from eis_toolkit.exceptions import MatchingRasterGridException, NonMatchingCrsException
from eis_toolkit.utilities.checks.raster import check_matching_crs


# The core snapping functionality. Used internally by snap.
def _snap(raster: rasterio.DatasetReader, snap_raster: rasterio.DatasetReader) -> Tuple[np.ndarray, dict]:
    raster_bounds = raster.bounds
    snap_bounds = snap_raster.bounds
    raster_pixel_size_x = raster.transform.a
    raster_pixel_size_y = abs(raster.transform.e)
    snap_pixel_size_x = snap_raster.transform.a
    snap_pixel_size_y = abs(snap_raster.transform.e)

    cells_added_x = ceil(snap_pixel_size_x / raster_pixel_size_x)
    cells_added_y = ceil(snap_pixel_size_y / raster_pixel_size_y)

    out_image = np.full((raster.count, raster.height + cells_added_y, raster.width + cells_added_x), raster.nodata)
    out_meta = raster.meta.copy()

    # Coordinates for the snap raster boundaries
    left_distance_in_pixels = (raster_bounds.left - snap_bounds.left) // snap_pixel_size_x
    left_snap_coordinate = snap_bounds.left + left_distance_in_pixels * snap_pixel_size_x

    bottom_distance_in_pixels = (raster_bounds.bottom - snap_bounds.bottom) // snap_pixel_size_y
    bottom_snap_coordinate = snap_bounds.bottom + bottom_distance_in_pixels * snap_pixel_size_y
    top_snap_coordinate = bottom_snap_coordinate + (raster.height + cells_added_y) * raster_pixel_size_y

    # Distance and array indices of close cell corner in snapped raster to slot values
    x_distance = (raster_bounds.left - left_snap_coordinate) % raster_pixel_size_x
    x0 = int((raster_bounds.left - left_snap_coordinate) // raster_pixel_size_x)
    x1 = x0 + raster.width

    y_distance = (raster_bounds.bottom - bottom_snap_coordinate) % raster_pixel_size_y
    y0 = int(cells_added_y - ((raster_bounds.bottom - bottom_snap_coordinate) // raster_pixel_size_y))
    y1 = y0 + raster.height

    # Find the closest corner of the snapped grid for shifting/slotting the original raster
    if x_distance < raster_pixel_size_x / 2 and y_distance < raster_pixel_size_y / 2:
        out_image[:, y0:y1, x0:x1] = raster.read()  # Snap values towards left-bottom
    elif x_distance < raster_pixel_size_x / 2 and y_distance > raster_pixel_size_y / 2:
        out_image[:, y0 - 1 : y1 - 1, x0:x1] = raster.read()  # Snap values towards left-top # noqa: E203
    elif x_distance > raster_pixel_size_x / 2 and y_distance > raster_pixel_size_y / 2:
        out_image[:, y0 - 1 : y1 - 1, x0 + 1 : x1 + 1] = raster.read()  # Snap values toward right-top # noqa: E203
    else:
        out_image[:, y0:y1, x0 + 1 : x1 + 1] = raster.read()  # Snap values towards right-bottom # noqa: E203

    out_transform = rasterio.Affine(
        raster.transform.a,
        raster.transform.b,
        left_snap_coordinate,
        raster.transform.d,
        raster.transform.e,
        top_snap_coordinate,
    )
    out_meta.update({"transform": out_transform, "width": out_image.shape[-1], "height": out_image.shape[-2]})
    return out_image, out_meta


@beartype
def snap_with_raster(raster: rasterio.DatasetReader, snap_raster: rasterio.DatasetReader) -> Tuple[np.ndarray, dict]:
    """Snaps/aligns raster to given snap raster.

    Raster is snapped from its left-bottom corner to nearest snap raster grid corner in left-bottom direction.
    If rasters are aligned, simply returns input raster data and metadata.

    Args:
        raster: The raster to be clipped.
        snap_raster: The snap raster i.e. reference grid raster.

    Returns:
        The snapped raster data.
        The updated metadata.

    Raises:
        NonMatchingCrsException: Raster and and snap raster are not in the same CRS.
        MatchingRasterGridException: Raster grids are already aligned.
    """

    if not check_matching_crs(
        objects=[raster, snap_raster],
    ):
        raise NonMatchingCrsException("Raster and and snap raster have different CRS.")

    if snap_raster.bounds.bottom == raster.bounds.bottom and snap_raster.bounds.left == raster.bounds.left:
        raise MatchingRasterGridException("Raster grids are already aligned.")

    out_image, out_meta = _snap(raster, snap_raster)
    return out_image, out_meta
