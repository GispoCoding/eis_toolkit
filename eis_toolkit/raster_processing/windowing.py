from typing import Tuple

import numpy as np
import rasterio
from rasterio import transform
from rasterio.windows import Window

from eis_toolkit.exceptions import CoordinatesOutOfBoundsException, InvalidWindowSizeException, NonMatchingCrsException


def _extract_window(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, center_x: float, center_y: float, window_size: int
) -> Tuple[np.ndarray, dict]:

    center_row, center_col = transform.rowcol(raster.transform, center_x, center_y)

    if window_size % 2 != 0:
        length_px = int(np.floor(window_size / 2))
        top_left_row = center_row - length_px
        top_left_col = center_col - length_px

    else:
        px_x, px_y = raster.transform * (center_col, center_row)
        length_px = int(window_size / 2)
        top_left_row = center_row - length_px
        top_left_col = center_col - length_px

        if center_x > px_x:
            top_left_col += 1
        if center_y > px_y:
            top_left_row += 1

    window = Window(
        col_off=top_left_col,
        row_off=top_left_row,
        width=window_size,
        height=window_size,
    )
    out_image = raster.read(
        boundless=True,
        window=window,
        fill_value=-9999,
    )

    top_left_coordinates = transform.xy(
        transform=raster.transform,
        rows=top_left_row,
        cols=top_left_col,
        offset="ul",
    )

    out_transform = rasterio.Affine(
        raster.transform[0],
        raster.transform[1],
        top_left_coordinates[0],
        raster.transform[3],
        raster.transform[4],
        top_left_coordinates[1],
    )

    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    return out_image, out_meta


def extract_window(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader, center_coords: Tuple[float, float], center_coord_crs: int, window_size: int
) -> Tuple[np.ndarray, dict]:
    """Extract window from raster.

       Center coordinate must be inside the raster but window can extent outside the raster in which case padding with
       -9999 is used.
    Args:
        raster (rasterio.io.DatasetReader): Source raster.
        center_coords (Tuple[int, int]): center coordinates for window int the form (x, y).
        center_coord_crs (int): EPSG code that defines the coordinate reference system.
        window_size (int): Side length of the rectangular window in pixels.
    Returns:
        out_image (numpy.ndarray): Extracted raster window.
        out_meta (dict): The updated metadata.

    Raises:
        CoordinatesOutOfBoundException: Window center coordinates are out of raster bounds.
        InvalidWindowSizeException: Window size is too small.
        NonMatchingCrsException: Raster and center coordinates are not in same crs.
    """

    center_x = center_coords[0]
    center_y = center_coords[1]

    if window_size < 1:
        raise InvalidWindowSizeException

    if (
        center_x < raster.bounds.left
        or center_x > raster.bounds.right
        or center_y < raster.bounds.bottom
        or center_y > raster.bounds.top
    ):
        raise CoordinatesOutOfBoundsException

    if center_coord_crs != int(raster.crs.to_string()[5:]):
        raise NonMatchingCrsException

    out_image, out_meta = _extract_window(raster, center_x, center_y, window_size)

    return out_image, out_meta
