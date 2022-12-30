"""
Created an November 24 2022
@author: torchala 
""" 

############ creates tiff-files from array

import numpy as np
import rasterio
from pathlib import Path
import os

parent_dir = Path(__file__).parent
outpath = parent_dir.joinpath(r'data')   # /test1.tif')
print (outpath)

#
arr1 = np.array([[5,8,4,5,6,3,0,3,4],
                   [8,8,2,9,8,5,6,3,4],
                   [0,9,2,3,0,6,8,7,5],
                   [5,1,1,4,3,6,0,2,8],
                   [9,0,4,6,8,0,1,2,4]])

#arr1 = np.random.randint(low = 0, high = 9, size = (100,200))

arr2 = np.array([[1,3,2,2,4,9,0,5,6],
                   [7,0,4,0,1,2,9,5,3],
                   [0,2,5,3,0,6,9,1,4],
                   [3,9,7,8,7,2,4,6,7],
                   [2,7,4,0,3,0,3,4,5]])

#arr2 = np.random.randint(low = 0, high = 9, size = (100,200))

arry = arr1+arr2          # kleines 1*1
#arry[2,2] = np.nan

# arry = np.array([[13,9,8,12,11,12,0,8,10],
#                  [15,8,6,9,6,7,14,8,9],
#                  [1,11,7,6,9,8,17,8,5],
#                  [8,10,8,12,10,8,4,8,13],
#                  [11,7,8,6,11,0,7,6,9]])
#arry = np.array([[0,0,0],[1, 1, 1], [2, 2, 2], [3 ,3 ,3]])
print (arr1)
print (arr2)

# profile={'affine': Affine(1.0, 0.0, 401900.0, 0.0, -1.0, 7620200.0),
# 'count': 1,  'crs': CRS({'init': 'epsg:32606'}),  'driver': 'GTiff',
# 'dtype': 'float32',  'height': 11100,  'interleave': 'band',
# 'nodata': -9999.0,  'tiled': False,
# 'transform': (401900.0, 1.0, 0.0, 7620200.0, 0.0, -1.0),  'width': 13750}

profile={
'count': 1,  'driver': 'GTiff',
'dtype': 'int8', #'float32',  
'tiled': False,
'width':9,  #100,
'height':5, #200
}

# profile.update(
#     dtype=rasterio.uint16,
#     count=1#,
#     #compress='lzw'
# )

with rasterio.open(os.path.join(outpath, 'test1k.tif'), 'w', **profile) as dst:
    dst.write_band(1, arr1.astype(rasterio.float64))
with rasterio.open(os.path.join(outpath, 'test2k.tif'), 'w', **profile) as dst:
    dst.write_band(1, arr2.astype(rasterio.float64))
with rasterio.open(os.path.join(outpath, 'summek.tif'), 'w', **profile) as dst:
    dst.write_band(1, arry.astype(rasterio.float64))

# profile.update(
#     dtype=rasterio.uint16#,
#     #count=1#,
#     #compress='lzw'
# )

# with rasterio.open(os.path.join(outpath, 'test2.tif'), 'w', **profile) as dst:
#     dst.write_band(1, arr1.astype(rasterio.float32))'

exit

