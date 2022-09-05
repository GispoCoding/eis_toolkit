import rasterio
import numpy as np
from rasterio import transform
from rasterio.windows import Window
from typing import Tuple

from eis_toolkit.exceptions import CoordinatesOutOfBoundExeption, InvalidWindowSizeException

def extract_window(
    raster: rasterio.io.DatasetReader,
    center_x: int,
    center_y: int,
    win_size: int
) -> Tuple[np.ndarray, dict]:
    """Extracts window from raster. Center coordinate must be inside the raster but
       window can extent outside the raster in which case padding with nodata value is
       used.
    Args:
        raster (rasterio.io.DatasetReader): Source raster.
        center_x (int): x coordinate for window center.
        center_y (int): y coordinate for window center.
        win_size (int): Side length of the rectangular window, must be odd number
            greater than 3.
    Returns:
        out_image (numpy.ndarray): Extracted raster window.
        out_meta (dict): The updated metadata.

    Raises:
        InvalidWindowSizeException: Window size is too small or it is not odd number.
        CoordinatesOutOfBoundException: Window center coordinates are out of raster
        bounds.
    """

    if win_size % 2 == 0 or win_size < 3:
        raise InvalidWindowSizeException

    if (
        center_x < raster.bounds.left or
        center_x > raster.bounds.right or
        center_y < raster.bounds.bottom or
        center_y > raster.bounds.top
    ):
        raise CoordinatesOutOfBoundExeption

     # Get the row and col location for center pixel
    row, col = transform.rowcol(
        raster.transform,
        center_x,
        center_y
    )
    
    # Get the offset for top left pixel
    px = int(np.floor(win_size / 2))
    tl_row = row - px
    tl_col = col - px

    # Read window
    window = Window(
        col_off=tl_col,
        row_off=tl_row,
        width=win_size,
        height=win_size
    )
    out_image = raster.read(
        boundless=True,
        window=window,
        fill_value=raster.nodata
    )

    # Get top left coordinates for the window
    tl = transform.xy(
        transform=raster.transform,
        rows=tl_row,
        cols=tl_col,
        offset='ul'
    )

    # Updated transformation matrix
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
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    return out_image, out_meta