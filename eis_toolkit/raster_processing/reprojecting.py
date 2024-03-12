import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Tuple
from rasterio import warp

from eis_toolkit.exceptions import MatchingCrsException
from eis_toolkit.raster_processing.resampling import RESAMPLE_METHOD_MAP


# Core reprojecting functionality used internally by reproject_raster and reproject_and_write_raster
def _reproject_raster(
    raster: rasterio.io.DatasetReader, target_crs: int, resampling_method: warp.Resampling
) -> Tuple[np.ndarray, dict]:

    src_arr = raster.read()
    dst_crs = rasterio.crs.CRS.from_epsg(target_crs)

    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
        raster.crs,
        dst_crs,
        raster.width,
        raster.height,
        *raster.bounds,
    )

    # Initialize output raster
    dst = np.empty((raster.count, dst_height, dst_width))
    dst.fill(raster.meta["nodata"])

    out_image = warp.reproject(
        source=src_arr,
        src_transform=raster.transform,
        src_crs=raster.crs,
        destination=dst,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=raster.meta["nodata"],
        dst_nodata=raster.meta["nodata"],
        resampling=resampling_method,
    )[0]

    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
        }
    )

    return out_image, out_meta


@beartype
def reproject_raster(
    raster: rasterio.io.DatasetReader,
    target_crs: int,
    resampling_method: Literal["nearest", "bilinear", "cubic", "average", "gauss", "max", "min"] = "nearest",
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match given coordinate reference system (EPSG).

    Args:
        raster: The raster to be reprojected.
        target_crs: Target CRS as EPSG code.
        resampling_method: Resampling method. Most suitable method depends on the dataset and context.
            Nearest, bilinear and cubic are some common choices. This parameter defaults to nearest.

    Returns:
        The reprojected raster data.
        The updated metadata.

    Raises:
        NonMatchinCrsException: Raster is already in the target CRS.
    """
    if target_crs == int(raster.crs.to_string()[5:]):
        raise MatchingCrsException("Raster is already in the target CRS.")

    method = RESAMPLE_METHOD_MAP[resampling_method]
    out_image, out_meta = _reproject_raster(raster, target_crs, method)

    return out_image, out_meta
