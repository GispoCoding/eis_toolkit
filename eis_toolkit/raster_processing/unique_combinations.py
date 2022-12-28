from typing import List, Dict

import itertools
import numpy as np
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _unique_combinations(  # type: ignore[no-any-unimported]
    list_of_rasters: List[rasterio.io.DatasetReader],
    path_to_output_file: str
) -> rasterio.io.DatasetReader:

    list_of_rasters = sorted(list_of_rasters, key=sort_by_raster_band_count)
    rasters = []
    for raster in list_of_rasters:
        for index in range(0, raster.count):
            band = index + 1
            rasters.append(raster.read(band))

    output_array = np.empty_like(rasters[0])
    combinations: Dict[tuple, int] = {}
    for row, column in itertools.product(range(len(rasters[0])), range(len(rasters[0][0]))):

        combination = tuple(rasters[raster][row][column] for raster, _ in enumerate(rasters))

        if combination not in combinations:
            combinations[combination] = len(combinations) + 1
        output_array[row][column] = combinations[combination]

    with rasterio.open(path_to_output_file, 'w', **list_of_rasters[0].meta) as write_raster:
        write_raster.write(output_array, 1)
    output_raster = rasterio.open(path_to_output_file)
    return output_raster

def sort_by_raster_band_count(raster):
    return raster.count


def unique_combinations(  # type: ignore[no-any-unimported]
    list_of_rasters: List[rasterio.io.DatasetReader],
    path_to_output_file: str
) -> rasterio.io.DatasetReader:
    """Get combinations of raster values between rasters.

    All bands in all rasters are used for analysis.

    Args:
        rasters (List[rasterio.io.DatasetReader]): Rasters to be used for finding combinations.
        path_to_output_file (str): The output file location and name for the output.

    Returns:
        rasterio.io.DatasetReader: Raster with unique combinations
    """
    rasters = []
    for raster in list_of_rasters:
        rasters.append(raster.read())

    row_length = len(rasters[0][0])
    column_length = len(rasters[0][0][0])

    if len(list_of_rasters) == 1:
        raise InvalidParameterValueException

    if not all(len(raster[0]) == row_length for raster in rasters):
        raise InvalidParameterValueException

    if not all(len(raster[0][0]) == column_length for raster in rasters):
        raise InvalidParameterValueException

    output_raster = _unique_combinations(list_of_rasters, path_to_output_file)
    return output_raster
