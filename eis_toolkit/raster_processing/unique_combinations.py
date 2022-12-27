from typing import List

import itertools
import numpy as np
import rasterio

#from eis_toolkit.exceptions import InvalidParameterValueException

def _unique_combinations(rasters: List[rasterio.io.DatasetReader], path_to_output_file) -> rasterio.io.DatasetReader:
    
    arrays=[]
    for raster in rasters:
        arrays.append(raster.read())
        
    output_array = np.empty_like(arrays[0])

    combinations = {}
    for row, column in itertools.product(range(len(arrays[0][0])), range(len(arrays[0][0][0]))):
        combination = tuple(arrays[i][0][row][column] for i, _ in enumerate(arrays))

        if combination not in combinations:
            combinations[combination] = len(combinations) + 1
        output_array[0][row][column] = combinations[combination]

    with rasterio.open(path_to_output_file, 'r+', **rasters[0].meta) as write_raster:
        write_raster.write(output_array)
        write_raster.close()
    with rasterio.open(path_to_output_file) as output_raster:
        return output_raster


def unique_combinations(rasters: List[rasterio.io.DatasetReader], path_to_output_file: str) -> rasterio.io.DatasetReader:

    arrays=[]
    for raster in rasters:
        arrays.append(raster.read())

    row_length = len(arrays[0][0])
    column_length = len(arrays[0][0][0])

    if not all(len(array[0]) == row_length for array in arrays):
        raise ValueError('not all rasters have same length!')

    if not all(len(array[0][0]) == column_length for array in arrays):
        raise ValueError('not all rasters have same length!')

    output_raster = _unique_combinations(rasters, path_to_output_file)
    return output_raster