from osgeo import gdal


def driver_cnt(a: int) -> None:
    """Tests whether it works to call one of gdal's functions.

    Args:
        a (int): input determining the functionality of the function
    """
    if a == 1:
        gdal.AllRegister()
        driver = gdal.GetDriverByName('GTiff')
        print('GeoTiffien kasittelyyn soveltuvia drivereita: ', driver.Register())
    else:
        print('Muuten vaan!')


driver_cnt(1)
