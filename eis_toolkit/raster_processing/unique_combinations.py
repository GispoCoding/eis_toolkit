import itertools
from typing import Dict, List, Tuple

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
    bands = []
    out_meta = raster_list[0].meta
    out_meta["count"] = 1
    height = raster_list[0].meta["height"]
    width = raster_list[0].meta["width"]

    for raster in raster_list:
        for band in range(1, raster.count + 1):
            bands.append(raster.read(band))

    if len(bands) == 1:
        raise InvalidParameterValueException

    # Check if all rasters have the same pixel width.
    if not all(raster.transform[0] == raster_list[0].transform[0] for raster in raster_list):
        raise InvalidParameterValueException

    # Check if all rasters have the same pixel height.
    if not all(raster.transform[4] == raster_list[0].transform[4] for raster in raster_list):
        raise InvalidParameterValueException

    if not all(raster.meta["crs"] == out_meta["crs"] for raster in raster_list):
        raise InvalidParameterValueException

    if not all(len(band) == height for band in bands):
        raise InvalidParameterValueException

    if not all(len(band[0]) == width for band in bands):
        raise InvalidParameterValueException

    out_image = _unique_combinations(bands)
    return out_image, out_meta
