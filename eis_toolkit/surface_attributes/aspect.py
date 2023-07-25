import rasterio
import numpy as np

from beartype import beartype
from beartype.typing import Optional, Union, Literal

from eis_toolkit.surface_attributes.partial_derivatives import method_horn
from eis_toolkit.utilities.nodata import nodata_to_nan, nan_to_nodata
from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import InvalidRasterBandException, NonSquarePixelSizeException


@beartype
def _get_aspect(
    raster: rasterio.io.DatasetReader, method: Literal, p: Union[np.ndarray, None], q: Union[np.ndarray, None]
) -> np.ndarray:

    cellsize = raster.res[0]
    out_meta = raster.meta.copy()
    out_array = np.squeeze(raster.read())

    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)

    if method == "Horn81" and any(coefficient is None for coefficient in (p, q)):
        p = method_horn(out_array, cellsize=cellsize, parameter="p")
        q = method_horn(out_array, cellsize=cellsize, parameter="q")

    out_array = (np.pi + np.arctan2(p, -q)) * (180.0 / np.pi)
    out_array = np.where(np.logical_and(p == 0, q == 0), -1, out_array)

    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata).astype(np.float16)
    out_meta.update({"dtype": out_array.dtype.name})

    return out_array, out_meta


def get_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal = "Horn81",
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate the aspect of a given surface.

    A raster cell with a slope of 0 essentially implies a flat surface
    with no inclination in any particular direction. Thus, for cells with a
    slope of 0, the aspect is not defined and set to -1.

    Args:
        raster: The input raster data.

    Returns:
        The calculated aspect in degree (0-360).
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if check_quadratic_pixels(raster) is False:
        raise NonSquarePixelSizeException("Processing requires quadratic pixel dimensions.")

    return _get_aspect(raster, method, p, q)
