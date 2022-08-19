from osgeo import gdal


def driver_cnt() -> None:
    """Tests whether it works to call one of gdal's functions."""
    gdal.AllRegister()
    driver = gdal.GetDriverByName('GTiff')
    print('Number of suitable drivers for geotiffs: ', driver.Register())
