from typing import List, Dict

import itertools
import numpy as np
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _unique_combinations(  # type: ignore[no-any-unimported]
    list_of_rasters: List[rasterio.io.DatasetReader],
    path_to_output_file: str
) -> rasterio.io.DatasetReader:

    bands = []
    for raster in list_of_rasters:
        bands.append(raster.read())

    output_band = np.empty_like(bands[0])
    combinations: Dict[tuple, int] = {}
    for row, column in itertools.product(range(len(bands[0])), range(len(bands[0][0]))):

        combination = tuple(bands[band][row][column] for band, _ in enumerate(bands))

        if combination not in combinations:
            combinations[combination] = len(combinations) + 1
        output_band[row][column] = combinations[combination]

    with rasterio.open(path_to_output_file, 'w', **list_of_rasters[0].meta) as write_raster:
        write_raster.write(output_band, 1)
    output_raster = rasterio.open(path_to_output_file)
    return output_raster


def unique_combinations(  # type: ignore[no-any-unimported]
    list_of_rasters: List[rasterio.io.DatasetReader],
    path_to_output_file: str
) -> rasterio.io.DatasetReader:
    """Get combinations of raster values between rasters.

    All bands in all rasters are used for analysis.
    A single band raster is used for reference when making the output.

    Args:
        list_of_rasters (List[rasterio.io.DatasetReader]): Rasters to be used for finding combinations.
        path_to_output_file (str): The output file location and name for the output.

    Returns:
        rasterio.io.DatasetReader: Raster with unique combinations
    """
    rasters = []
    for raster in list_of_rasters:
        rasters.append(raster.read())

    row_length = len(rasters[0][0])
    column_length = len(rasters[0][0][0])

    if len(rasters) == 1:
        raise InvalidParameterValueException

    if not all(len(raster[0]) == row_length for raster in rasters):
        raise InvalidParameterValueException

    if not all(len(raster[0][0]) == column_length for raster in rasters):
        raise InvalidParameterValueException

    output_raster = _unique_combinations(list_of_rasters, path_to_output_file)
    return output_raster
