import rasterio
import numpy as np
from rasterio import transform
from typing import Tuple

from eis_toolkit.exceptions import CoordinatesOutOfBoundExeption, InvalidWindowSizeException

def extract_window(
    raster: rasterio.io.DatasetReader,
    center_x: int,
    center_y: int,
    win_size: int
) -> Tuple[np.ndarray, dict]:
    """Extracts window from raster.
    Args:
        raster (rasterio.io.DatasetReader): Source raster.
        center_x (int): x coordinate for window center.
        center_y (int): y coordinate for window center.
        window_size (int): Side length of the rectangular window, must be odd number
            greater than 3.
    Returns:
        out_image (numpy.ndarray): Extracted raster window.
        out_meta (dict): The updated metadata.

    Raises:
        InvalidWindowSizeException: Window size is too small or it is not odd number.
        CoordinatesOutOfBoundException: Window center coordinates are out of raster
            bounds.
    """

    # deal edge cases/padding

    if win_size % 2 == 0 or win_size < 3:
        raise InvalidWindowSizeException

    if (
        center_x < raster.bounds.left or
        center_x > raster.bounds.right or
        center_y < raster.bounds.bottom or
        center_y > raster.bounds.top
    ):
        raise CoordinatesOutOfBoundExeption

    row, col = transform.rowcol(
        raster.transform,
        center_x,
        center_y
    )

    px = int(np.floor(win_size / 2))
    raster_arr = raster.read()
    out_image = raster_arr[:,row-px:row+px+1, col-px:col+px+1]

    tl = transform.xy(
        transform=raster.transform,
        rows=row-px,
        cols=col-px,
        offset='ul'
    )

    out_transform = rasterio.Affine(
        raster.transform[0],
        raster.transform[1],
        tl[0],
        raster.transform[3],
        raster.transform[4],
        tl[1]
    )

    out_meta = raster.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    return out_image, out_meta



rast = rasterio.open("tests/data/remote/small_raster.tif")
rast.bounds

win = extract_window(rast, 384800, 6671280, 5)

