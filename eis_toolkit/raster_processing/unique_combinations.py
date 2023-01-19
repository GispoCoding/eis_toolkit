from typing import List, Dict, Tuple

import itertools
import numpy as np
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _unique_combinations(
    bands: List[np.ndarray],
) -> np.ndarray:

    out_image = np.empty_like(bands[0])
    combinations: Dict[tuple, int] = {}
    for row, column in itertools.product(range(len(bands[0])), range(len(bands[0][0]))):

        combination = tuple(bands[band][row][column] for band, _ in enumerate(bands))

        if combination not in combinations:
            combinations[combination] = len(combinations) + 1
        out_image[row][column] = combinations[combination]

    return out_image


def unique_combinations(  # type: ignore[no-any-unimported]
    raster_list: List[rasterio.io.DatasetReader],
) -> Tuple[np.ndarray, dict]:
    """Get combinations of raster values between rasters.

    All bands in all rasters are used for analysis.
    A single band raster is used for reference when making the output.

    Args:
        raster_list (List[rasterio.io.DatasetReader]): Rasters to be used for finding combinations.

    Returns:
        out_image (numpy.ndarray): Combinations of rasters.
        out_meta (dict): The metadata of the first raster in raster_list.
    """
    rasters = []
    out_meta = raster_list[0].meta
    out_meta['count'] = 1
    height = raster_list[0].meta['height']
    width = raster_list[0].meta['width']

    for raster in raster_list:
        for band in range(1, raster.count+1):
            rasters.append(raster.read(band))

    if len(rasters) == 1:
        raise InvalidParameterValueException

    if not all(len(raster) == height for raster in rasters):
        raise InvalidParameterValueException

    if not all(len(raster[0]) == width for raster in rasters):
        raise InvalidParameterValueException

    out_image = _unique_combinations(rasters)
    return out_image, out_meta
