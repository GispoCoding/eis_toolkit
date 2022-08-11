import geopandas
import rasterio
from rasterio.mask import mask

from eis_toolkit.checks.crs import check_crs_matches
from eis_toolkit.checks.geometry import check_geometry_types


def clip(
    raster: rasterio.io.DatasetReader,
    polygon: geopandas.GeoDataFrame,
):
    """Clips a raster with a polygon.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        polygon (geopandas.GeoDataFrame): A polygon geodataframe to do the
        clipping with.
    Returns:
        out_image ():
        out_transform ():
        out_meta ():
    """

    check_crs_matches([raster, polygon])
    shapes = polygon["geometry"].to_list()
    check_geometry_types(
        shapes,
        allowed_types=['Polygon', 'MultiPolygon']
    )
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
    # TODO return these, or: Write to tempfile -> read it -> return opened
    # raster?
    return out_image, out_transform, out_meta
