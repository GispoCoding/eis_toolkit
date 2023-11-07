import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import List, Tuple

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.check_raster_grids import check_raster_grids


def _unique_combinations(
    bands: List[np.ndarray],
) -> np.ndarray:

    combined_array = np.stack(bands, axis=-1)

    combined_array_2d = combined_array.reshape(-1, len(bands))

    _, unique_indices, inverse_indices = np.unique(combined_array_2d, axis=0, return_index=True, return_inverse=True)

    unique_combinations = inverse_indices.reshape(bands[0].shape)

    unique_combinations += 1

    return unique_combinations


@beartype
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

    for raster in raster_list:
        for band in range(1, raster.count + 1):
            bands.append(raster.read(band))

    if len(bands) == 1:
        raise InvalidParameterValueException("Expected to have more bands than 1")

    if check_raster_grids(raster_list) is not True:
        raise InvalidParameterValueException("Expected raster grids to be of same shape")

    out_image = _unique_combinations(bands)
    return out_image, out_meta
