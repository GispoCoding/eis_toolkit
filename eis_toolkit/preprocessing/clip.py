import numpy as np
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.mask import mask
from affine import Affine
from typing import Tuple
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def clip_ras(rasin: Path, polin: Path) -> Tuple[np.ndarray, Affine]:
    """Clips raster with polygon.

    Args:
        rasin (Path): file path to input raster
        polin (Path): file path to polygon to be used for clipping the input raster

    Returns:
        Tuple[np.ndarray, Affine]: tuple consisting of clipped raster in array format and georeferencing information
    """
    pol_df = gpd.read_file(polin)
    with rasterio.open(rasin) as src:
        pol_geom = pol_df['geometry'][0]
        if pol_geom.geom_type not in ['Polygon', 'MultiPolygon']:
            raise(NotApplicableGeometryTypeException)
        if src.crs != pol_df.crs:
            raise(NonMatchingCrsException)
        out_image, out_transform = mask(src, [pol_geom], crop=True)

    return out_image, out_transform
