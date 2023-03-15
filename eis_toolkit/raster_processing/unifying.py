from typing import List, Tuple

import numpy as np
import rasterio
import geopandas as gpd

from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio import warp
from shapely.geometry import box

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing import reprojecting, resampling, snapping, clipping


# The core unifying functionality. Used internally by unify_rasters.
def _unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    raster_list: List[rasterio.io.DatasetReader],
    resampling_method: Resampling,
    same_extent: bool
) -> List[Tuple[np.ndarray, dict]]:

    target_epsg = int(base_raster.crs.to_string()[5:])

    base_image = base_raster.read()
    base_meta = base_raster.meta.copy()
    out_rasters = [(base_image, base_meta)]

    for raster in raster_list:
        # Reproject
        if raster.crs != base_raster.crs:
            out_image, out_meta = reprojecting.reproject_raster(raster, target_epsg, resampling_method)
        else:
            out_image, out_meta = raster.read(), raster.meta.copy()

        # Save to memory, then resample
        upscale_factor_x = out_meta["transform"].a / base_raster.transform.a
        upscale_factor_y = out_meta["transform"].e / base_raster.transform.e

        if upscale_factor_x != 1 or upscale_factor_y != 1:
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as dataset:
                    dataset.write(out_image)
                # with memfile.open() as raster:
                    out_image, out_meta = resampling.resample(
                        dataset, upscale_factor_x, upscale_factor_y, resampling_method
                    )
                    # transform = rasterio.Affine(
                    #     base_raster.transform.a,
                    #     out_meta['transform'].b,
                    #     out_meta['transform'].c,
                    #     out_meta['transform'].d,
                    #     base_raster.transform.e,
                    #     out_meta['transform'].f
                    # )
                    # out_meta['transform'] = transform

        # print(out_meta['transform'], out_meta['width'], out_meta['height'])
        
        # trans, width, height = warp.aligned_target(
        #     out_meta['transform'],
        #     out_meta['width'],
        #     out_meta['height'],
        #     (base_raster.transform.a, abs(base_raster.transform.e))
        # )


        # print("Aligned target:")
        # print(trans, width, height)   
        # raise Exception

        # Save to memory, then snap
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_image)
            # with memfile.open() as raster:
                out_image, out_meta = snapping.snap_with_raster(dataset, base_raster)

        # Save to memory, then clip for same extent
        if same_extent:
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as dataset:
                    dataset.write(out_image)
                # with memfile.open() as raster:
                    geo = gpd.GeoDataFrame(
                        {"geometry": box(*base_raster.bounds)},
                        index = [0],
                        crs = base_raster.crs.to_epsg()
                    )
                    out_image, out_meta = clipping.clip_raster(dataset, geo)

        out_rasters.append((out_image, out_meta))

    return out_rasters


def unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    raster_list: List[rasterio.io.DatasetReader],
    resampling_method: Resampling = Resampling.bilinear,
    same_extent: bool = False
) -> List[Tuple[np.ndarray, dict]]:
    """Unifies (reprojects, resamples and aligns) given rasters relative to a base raster.

    Args:
        raster (rasterio.io.DatasetReader): The base raster to determine unifying.
        raster_list (list(rasterio.io.DatasetReader)): List of rasters to be unified.
        resampling_method (rasterio.enums.Resampling): Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.
        same_extent (bool): If the result rasters need to have same extent/bounds as the
            base raster.

    Returns:
        out_rasters (list(tuple(numpy.ndarray, dict))): List of unified rasters' data and metadata.
            First element is the base raster.

    Raises:
        InvalidParameterValueException: Upscale factor is not a positive value.
    """
    if not isinstance(base_raster, rasterio.io.DatasetReader):
        raise InvalidParameterValueException
    if not isinstance(raster_list, list):
        raise InvalidParameterValueException
    if not all(isinstance(raster, rasterio.io.DatasetReader) for raster in raster_list):
        raise InvalidParameterValueException
    if len(raster_list) == 0:
        raise InvalidParameterValueException

    out_rasters = _unify_rasters(base_raster, raster_list, resampling_method, same_extent)
    return out_rasters
