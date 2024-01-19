import geopandas
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Sequence, Tuple
from rasterio.mask import mask

from eis_toolkit.exceptions import GeometryTypeException, NonMatchingCrsException
from eis_toolkit.utilities.checks.geometry import check_geometry_types
from eis_toolkit.utilities.checks.raster import check_matching_crs


# The core clipping functionality. Used internally by clip.
def _clip_raster(raster: rasterio.io.DatasetReader, geometries: Sequence) -> Tuple[np.ndarray, dict]:
    out_image, out_transform = mask(dataset=raster, shapes=geometries, crop=True, all_touched=True)
    out_meta = raster.meta.copy()
    out_meta.update(
        {"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
    )

    return out_image, out_meta


@beartype
def clip_raster(raster: rasterio.io.DatasetReader, geodataframe: geopandas.GeoDataFrame) -> Tuple[np.ndarray, dict]:
    """Clips a raster with polygon geometries.

    Args:
        raster: The raster to be clipped.
        geodataframe: A geodataframe containing the geometries to do the clipping with.
            Should contain only polygon features.

    Returns:
        The clipped raster data.
        The updated metadata.

    Raises:
        NonMatchingCrsException: The raster and geodataframe are not in the same CRS.
        GeometryTypeException: The input geometries contain non-polygon features.
    """
    geometries = geodataframe["geometry"]

    if not check_matching_crs(
        objects=[raster, geometries],
    ):
        raise NonMatchingCrsException("The raster and geodataframe are not in the same CRS.")

    if not check_geometry_types(
        geometries=geometries,
        allowed_types=["Polygon", "MultiPolygon"],
    ):
        raise GeometryTypeException("The input geometries contain non-polygon features.")

    out_image, out_meta = _clip_raster(
        raster=raster,
        geometries=geometries,
    )

    return out_image, out_meta
