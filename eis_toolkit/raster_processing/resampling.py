from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Tuple
from rasterio import warp
from rasterio.enums import Resampling

from eis_toolkit.exceptions import NumericValueSignException

RESAMPLING_METHODS = {
    "nearest": warp.Resampling.nearest,
    "bilinear": warp.Resampling.bilinear,
    "cubic": warp.Resampling.cubic,
    "cubic_spline": warp.Resampling.cubic_spline,
    "lanczos": warp.Resampling.lanczos,
    "average": warp.Resampling.average,
    "mode": warp.Resampling.mode,
    # "gauss": warp.Resampling.gauss,  # Not available for rasterio.warp apparently
    "max": warp.Resampling.max,
    "min": warp.Resampling.min,
    "median": warp.Resampling.med,
    "q1": warp.Resampling.q1,
    "q3": warp.Resampling.q3,
    "sum": warp.Resampling.sum,
    "rms": warp.Resampling.rms,
}


def _resample(
    raster: rasterio.io.DatasetReader, resolution: Number, resampling_method: Resampling
) -> Tuple[np.ndarray, dict]:

    resolution = float(resolution)

    dst_height, dst_width = (
        int(raster.height * raster.res[0] / resolution),
        int(raster.width * raster.res[1] / resolution),
    )
    out_transform = rasterio.Affine(resolution, 0, raster.transform[2], 0, -resolution, raster.transform[5])

    dst = np.empty((raster.count, dst_height, dst_width))
    dst.fill(raster.meta["nodata"])

    out_image = warp.reproject(
        source=raster.read(),
        destination=dst,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=out_transform,
        dst_crs=raster.crs,
        src_nodata=raster.meta["nodata"],
        dst_nodata=raster.meta["nodata"],
        resampling=resampling_method,
    )

    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "transform": out_transform,
            "width": out_image[0].shape[-1],
            "height": out_image[0].shape[-2],
        }
    )

    return out_image[0], out_meta


@beartype
def resample(
    raster: rasterio.io.DatasetReader,
    resolution: Number,
    resampling_method: Literal[
        "nearest",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "median",
        "q1",
        "q3",
        "sum",
        "rms",
    ] = "bilinear",
) -> Tuple[np.ndarray, dict]:
    """Resamples raster according to given resolution.

    Args:
        raster: The raster to be resampled.
        resolution: Target resolution i.e. cell size of the output raster.
        resampling_method: Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.

    Returns:
        The resampled raster data.
        The updated metadata.

    Raises:
        NumericValueSignException: Resolution is not a positive value.
    """
    if resolution <= 0:
        raise NumericValueSignException(f"Expected a positive value for resolution: {resolution})")

    method = RESAMPLING_METHODS[resampling_method]
    out_image, out_meta = _resample(raster, resolution, method)
    return out_image, out_meta
