import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Sequence, Tuple

from eis_toolkit.exceptions import InvalidParameterValueException, NonMatchingRasterMetadataException
from eis_toolkit.utilities.checks.raster import check_raster_grids


def _unique_combinations(
    bands: Sequence[np.ndarray],
) -> np.ndarray:

    combined_array = np.stack(bands, axis=-1)

    combined_array_2d = combined_array.reshape(-1, len(bands))

    _, unique_indices, inverse_indices = np.unique(combined_array_2d, axis=0, return_index=True, return_inverse=True)

    unique_combinations = inverse_indices.reshape(bands[0].shape)

    unique_combinations += 1

    return unique_combinations


@beartype
def unique_combinations(  # type: ignore[no-any-unimported]
    raster_list: Sequence[rasterio.io.DatasetReader],
) -> Tuple[np.ndarray, dict]:
    """Get combinations of raster values between rasters.

    All bands in all rasters are used for analysis.
    The first band of the first raster is used for reference when making the output.

    Args:
        raster_list: Rasters to be used for finding combinations.

    Returns:
        Combinations of rasters.
        The metadata of the first raster in raster_list.

    Raises:
        InvalidParameterValueException: Input rasters don't have enough bands to perform
            the operation or input rasters are of different shape.
    """
    bands = []
    out_meta = raster_list[0].meta
    out_meta["count"] = 1

    raster_profiles = []
    for raster in raster_list:
        for band in range(1, raster.count + 1):
            bands.append(raster.read(band))
        raster_profiles.append(raster.profile)

    if len(bands) == 1:
        raise InvalidParameterValueException("Expected to have more bands than 1")

    if check_raster_grids(raster_profiles) is not True:
        raise NonMatchingRasterMetadataException("Expected raster grids to be have the same grid properties.")

    out_image = _unique_combinations(bands)
    return out_image, out_meta
