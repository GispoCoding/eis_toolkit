from typing import Iterable
import geopandas
import rasterio
from rasterio.mask import mask

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.geometry import check_geometry_types
from eis_toolkit.exceptions import NotApplicableGeometryTypeException
from eis_toolkit.exceptions import NonMatchingCrsException


def __clip(
    raster: rasterio.io.DatasetReader,
    shapes: Iterable
):
    """The core clipping functionality to be used by clip"""

    out_image, out_transform = mask(
        dataset=raster,
        shapes=shapes,
        crop=True,
        all_touched=True
    )
    out_meta = raster.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })
    return out_image, out_meta


def clip(
    raster: rasterio.io.DatasetReader,
    polygon: geopandas.GeoDataFrame,
):
    """Clips a raster with vector geometry (polygon / polygons).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        polygon (geopandas.GeoDataFrame): A geodataframe containing the
            polygon(s) to do the clipping with.

    Returns:
        out_image (numpy.ndarray): The clipped raster data
        out_meta (dict): The updated metadata

    Raises:
        NonMatchingCrsException: The raster and polygons are not in the same
            crs
        NotApplicableGeometryTypeException: The input geometries contain
            non-polygon features
    """

    shapes = polygon["geometry"]
    if not check_matching_crs([raster, shapes]):
        raise NonMatchingCrsException
    if not check_geometry_types(shapes, allowed=["Polygon", "MultiPolygon"]):
        raise NotApplicableGeometryTypeException
    return __clip(raster=raster, shapes=shapes)
