import rasterio
import numpy as np
from rasterio import warp
from typing import Tuple

def reproject_raster(
    src: rasterio.io.DatasetReader,
    target_EPSG: int,
    resampling_method: str = "nearest"
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match given coordinate system (EPSG).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        target_EPSG (int): Target crs as EPSG code.
        resampling_method (str): Resampling method. Can be either 'nearest', 'bilinear'
            or 'cubic'

    Returns:
        out_image (numpy.ndarray): Reprojected raster data.
        out_meta (dict): The updated metadata.
    """

    resamplers =  {
        'nearest': warp.Resampling.nearest,
        'bilinear': warp.Resampling.bilinear,
        'cubic': warp.Resampling.cubic
    }

    # Compare input and target crs and do nothing if they match
    if target_EPSG == int(src.crs.to_string()[5:]):
        out_image = src.read()
        out_meta = src.meta.copy()
    
    # Else reproject the input raster
    else:
        src_arr = src.read()
        dst_crs = rasterio.CRS.from_epsg(target_EPSG)

        dst_transform, dst_width, dst_height = warp.calculate_default_transform(
           src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Initialize base raster (target raster)
        dst = np.empty((src.count, dst_height, dst_width))
        dst.fill(-9999)
        
        out_image = warp.reproject(
            source=src_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            destination=dst,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=-9999,
            dst_nodata=np.nan,
            resampling=resamplers[resampling_method]
        )[0]

        out_meta = src.meta.copy()
        out_meta.update({
            'crs': dst_crs,
            'transform': dst_transform,
            "width": dst_width,
            "height": dst_height,
        })

    return out_image, out_meta


def reproject_and_write_raster(
    src: rasterio.io.DatasetReader,
    target_EPSG: int,
    output_fp: str,
    resampling_method: str = "nearest"
) -> None:
    """Reprojects raster to match given coordinate system (EPSG) and saves it.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        target_EPSG (int): Target crs as EPSG code.
        output_fp (str): File path for reprojected raster.
        resampling_method (str): Resampling method. Can be either 'nearest', 'bilinear'
            or 'cubic'
    """

    resamplers =  {
        'nearest': warp.Resampling.nearest,
        'bilinear': warp.Resampling.bilinear,
        'cubic': warp.Resampling.cubic
    }

    dst_crs = rasterio.CRS.from_epsg(target_EPSG)

    transform, width, height = warp.calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)

    out_meta = src.meta.copy()
    out_meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(output_fp, 'w', **out_meta) as dst:
        warp.reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resamplers[resampling_method]
        )


def run_reproject():
    run_original = True
    input_fp = "tests/data/remote/small_raster.tif"
    output_fp = "tests/data/remote/output_raster1.tif"
    src_raster = rasterio.open(input_fp)

    if run_original:
        out_image, out_meta = reproject_raster(src_raster, 4326)
        with rasterio.open(output_fp, 'w', **out_meta) as output_dataset:
            output_dataset.write(out_image.astype(rasterio.float32))

    else:
        reproject_and_write_raster(src_raster, 4326, output_fp)
    

run_reproject()