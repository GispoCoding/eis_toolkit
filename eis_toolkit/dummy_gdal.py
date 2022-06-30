from osgeo import gdal

gdal.AllRegister()
driver = gdal.GetDriverByName('GTiff')
print('GeoTiffien kasittelyyn soveltuvia drivereita: ', driver.Register())
