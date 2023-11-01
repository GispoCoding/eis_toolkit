from typing import Dict, List, Tuple

import numpy as np
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _unique_combinations(
    bands: List[np.ndarray],
) -> np.ndarray:

    unique_combinations: Dict[int, int] = {}
    unique_indices = np.zeros(bands[0].shape, dtype=int)
    combination = 1

    for band in bands:
        for row in range(band.shape[0]):
            for column in range(band.shape[1]):
                raster_value = band[row, column]
                if raster_value not in unique_combinations:
                    unique_combinations[raster_value] = combination
                    combination += 1
                unique_indices[row, column] = unique_combinations[raster_value]

    return unique_indices


def unique_combinations(  # type: ignore[no-any-unimported]
    raster_list: List[rasterio.io.DatasetReader],
) -> Tuple[np.ndarray, dict]:
    """Get combinations of raster values between rasters.

    All bands in all rasters are used for analysis.
    The first band of the first raster is used for reference when making the output.

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
        raise InvalidParameterValueException("Expected to have more bands than 1")

    # Check if all rasters have the same pixel width.
    if not all(raster.transform[0] == raster_list[0].transform[0] for raster in raster_list):
        raise InvalidParameterValueException("Expected all raster files to have the same pixel width")

    # Check if all rasters have the same pixel height.
    if not all(raster.transform[4] == raster_list[0].transform[4] for raster in raster_list):
        raise InvalidParameterValueException("Expected all raster files to have the same pixel height")

    if not all(raster.meta["crs"] == out_meta["crs"] for raster in raster_list):
        raise InvalidParameterValueException("Expected all raster files to have the same coordinate system")

    if not all(len(band) == height for band in bands):
        raise InvalidParameterValueException("Expected all raster files to have the same height")

    if not all(len(band[0]) == width for band in bands):
        raise InvalidParameterValueException("Expected all raster files to have the same width")

    out_image = _unique_combinations(bands)
    return out_image, out_meta
