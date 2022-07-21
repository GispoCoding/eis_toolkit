import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.mask import mask
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def clip_ras(rasin: Path, polin: Path, rasout: Path) -> None:
    """Clips raster with polygon and saves resulting raster into given folder location.

    Args:
        rasin (Path): file path to input raster
        polin (Path): file path to polygon to be used for clipping the input raster
        rasout (Path): file path to output raster
    """
    pol_df = gpd.read_file(polin)

    with rasterio.open(rasin) as src:
        pol_geom = pol_df['geometry'][0]
        if pol_geom.geom_type not in ['Polygon', 'MultiPolygon']:
            raise (NotApplicableGeometryTypeException)
        if src.crs != pol_df.crs:
            raise (NonMatchingCrsException)
        out_image, out_transform = mask(src, [pol_geom], crop=True)
        out_meta = src.meta

    out_meta.update({'driver': 'GTiff',
                     'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})

    with rasterio.open(rasout, 'w', **out_meta) as dest:
        dest.write(out_image)

    return
