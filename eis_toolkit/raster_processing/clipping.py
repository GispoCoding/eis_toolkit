import geopandas
import rasterio
from rasterio.mask import mask

from eis_toolkit.checks.crs import check_crs
from eis_toolkit.checks.geometry import check_geometry_type


# TODO
def _extract_single_geometry(gdf: geopandas.GeoDataFrame):

    geometry = gdf['geometry'][0]
    return geometry


def clip(
    input_raster: rasterio.io.DatasetReader,
    input_polygon: geopandas.GeoDataFrame,
):
    """TODO Clips raster with polygon and saves resulting raster into given folder location.

    Args:
        rasin (Path): file path to input raster
        polin (Path): file path to polygon to be used for clipping the input raster
        rasout (Path): file path to output raster
    """

    check_crs([input_raster, input_polygon])
    polygon_geometry = _extract_single_geometry(input_polygon)
    check_geometry_type(
        polygon_geometry,
        allowed_types=['Polygon', 'MultiPolygon']
    )

    out_image, out_transform = mask(
        input_raster,
        [polygon_geometry],
        crop=True,
        all_touched=True
    )
    out_meta = input_raster.meta.copy()

    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    return out_image, out_transform, out_meta
    # TODO what do we want to return?
    # with rasterio.open(output_raster, 'w', **out_meta) as dest:
    #     dest.write(out_image)
