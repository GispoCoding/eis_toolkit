import geopandas as gpd
from pathlib import Path
from osgeo import gdal
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def clip_ras(rasin: str, polin: Path, rasout: str) -> gdal.Dataset:
    """Clips raster with polygon.

    Args:
        rasin (str): file path to input raster
        polin (Path): file path to polygon to be used for clipping the input raster
        rasout (str): file path to output raster

    Returns:
        gdal.Dataset: clipped raster
    """
    pol_df = gpd.read_file(polin)

    pol_geom = pol_df['geometry'][0]
    if pol_geom.geom_type not in ['Polygon', 'MultiPolygon']:
        raise (NotApplicableGeometryTypeException)
    if gdal.Open(rasin).GetProjection() != pol_df.crs:
        raise (NonMatchingCrsException)

    gdal.Warp(rasout, rasin, cutlineDSName=polin, cropToCutline=True)
    res = gdal.Open(rasout)

    return res
