from typing import Iterable, Tuple

import geopandas
import numpy as np
import rasterio
from rasterio.mask import mask

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.geometry import check_geometry_types
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


# The core clipping functionality. Used internally by clip.
def _clip(  # type: ignore[no-any-unimported] # noqa: E261
    raster: rasterio.io.DatasetReader, geometries: Iterable
) -> Tuple[np.ndarray, dict]:
    out_image, out_transform = mask(dataset=raster, shapes=geometries, crop=True, all_touched=True)
    out_meta = raster.meta.copy()
    out_meta.update(
        {"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
    )

    return out_image, out_meta


def clip(  # type: ignore[no-any-unimported] # noqa: E261,E262
    raster: rasterio.io.DatasetReader, geodataframe: geopandas.GeoDataFrame
) -> Tuple[np.ndarray, dict]:
    """Clips a raster with a geodataframe.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        geodataframe (geopandas.GeoDataFrame): A geodataframe containing the geometries to do the clipping with.

    Returns:
        out_image (np.ndarray): The clipped raster data.
        out_meta (dict): The updated metadata.

    Raises:
        NonMatchingCrsException: The raster and polygons are not in the same crs.
        NotApplicableGeometryTypeException: The input geometries contain non-polygon features.
    """
    geometries = geodataframe["geometry"]

    if not check_matching_crs(
        objects=[raster, geometries],
    ):
        raise NonMatchingCrsException

    if not check_geometry_types(
        geometries=geometries,
        allowed_types=["Polygon", "MultiPolygon"],
    ):
        raise NotApplicableGeometryTypeException

    out_image, out_meta = _clip(
        raster=raster,
        geometries=geometries,
    )

    return out_image, out_meta
