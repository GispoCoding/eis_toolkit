import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Sequence, Tuple
from rasterio import profiles

from eis_toolkit.exceptions import (
    InvalidDataShapeException,
    InvalidParameterValueException,
    NonMatchingRasterMetadataException,
)
from eis_toolkit.utilities.checks.raster import check_raster_grids


@beartype
def split_raster_bands(raster: rasterio.io.DatasetReader) -> Sequence[Tuple[np.ndarray, profiles.Profile]]:
    """
    Split multiband raster into singleband rasters.

    Args:
        raster: Input multiband raster.

    Returns:
        Output singleband raster list. List elements are tuples where first element is raster data and second \
        element is raster profile.

    Raises:
        InvalidParameterValueException: Input raster contains only one band.
    """
    out_rasters = []
    count = raster.meta["count"]
    if count < 2:
        raise InvalidParameterValueException(f"Expected multiple bands in the input raster. Bands: {count}.")

    for i in range(1, count + 1):
        band_data = raster.read(i)
        band_profile = raster.profile.copy()
        band_profile.update({"count": 1, "dtype": band_data.dtype})
        out_rasters.append((band_data, band_profile))
    return out_rasters


@beartype
def combine_raster_bands(input_rasters: Sequence[rasterio.io.DatasetReader]) -> Tuple[np.ndarray, profiles.Profile]:
    """
    Combine multiple rasters into one multiband raster.

    The input rasters can be either singleband or multiband. All bands are stacked in the order they are
    extracted from the input raster list.

    All input rasters must have matching spatial metadata (extent, pixel size, CRS).

    Args:
        input_rasters: List of rasters to combine.

    Returns:
        The combined raster data.
        The updated raster profile.

    Raises:
        InvalidParameterValueException: Input rasters contains only one raster.
        NonMatchingRasterMetadataException: Input rasters have mismatching spatial metadata.
    """
    if len(input_rasters) < 2:
        raise InvalidParameterValueException(
            f"Expected multiple rasters in the input_rasters list. Rasters: {len(input_rasters)}."
        )

    profiles = []
    bands_arrays = []
    for raster in input_rasters:
        profiles.append(raster.profile)
        for i in range(1, raster.count + 1):
            bands_arrays.append(raster.read(i))

    if not check_raster_grids(profiles, same_extent=True):
        raise NonMatchingRasterMetadataException("Input rasters have mismatching metadata/profiles.")

    out_image = np.stack(bands_arrays, axis=0)

    out_profile = profiles[0].copy()
    out_profile["count"] = len(bands_arrays)

    return out_image, out_profile


@beartype
def stack_raster_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """
    Stack 2D and 3D NumPy arrays (each representing a raster with one or multiple bands) along the bands axis.

    Parameters:
        arrays: List of 2D and 3D NumPy arrays. Each 2D array should have shape (height, width).
            and 3D array shape (bands, height, width).

    Returns:
        A single 3D NumPy array where the first dimension size equals the total number of bands.

    Raises:
        InvalidDataShapeException: Input raster arrays have mismatching shapes or all input rasters are not 2D or 3D.
    """
    processed_arrays = []
    for array in arrays:
        # Add a new axis if the array is 2D
        if array.ndim == 2:
            array = array[np.newaxis, :]
        elif array.ndim != 3:
            raise InvalidDataShapeException("All raster arrays must be 2D or 3D for stacking.")
        processed_arrays.append(array)

    shape_set = {arr.shape[1:] for arr in processed_arrays}
    if len(shape_set) != 1:
        raise InvalidDataShapeException(
            "All raster arrays must have the same shape in 2 last dimensions (height, width)."
        )

    # Stack along the first axis
    stacked_array = np.concatenate(processed_arrays, axis=0)

    return stacked_array
