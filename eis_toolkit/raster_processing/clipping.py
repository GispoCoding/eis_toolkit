import geopandas
import rasterio
from rasterio.mask import mask

from eis_toolkit.checks.crs import matching_crs
from eis_toolkit.checks.geometry import correct_geometry_types
from eis_toolkit.exceptions import NotApplicableGeometryTypeException
from eis_toolkit.exceptions import NonMatchingCrsException


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

    if not matching_crs([raster, shapes]):
        raise NonMatchingCrsException
    if not correct_geometry_types(shapes, allowed=["Polygon", "MultiPolygon"]):
        raise NotApplicableGeometryTypeException

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
