import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Tuple

from eis_toolkit.raster_processing.unifying import unify_raster_grids
from eis_toolkit.utilities.checks.raster import check_raster_grids


@beartype
def mask_raster(
    raster: rasterio.io.DatasetReader,
    base_raster: rasterio.io.DatasetReader,
) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    """
    Mask input raster using the nodata locations from base raster.

    Only the first band of base raster is used to scan for nodata cells. Masking is performed to all
    bands of input raster.

    If input rasters have mismatching grid properties, unifies rasters before masking (uses `nearest`
    resampling, unify separately first if you need control over the resampling method).

    Args:
        raster: The raster to be masked.
        base_raster: The base raster used to determine nodata locations.

    Returns:
        The masked raster data.
        The raster profile.
    """
    raster_profile = raster.profile
    base_raster_profile = base_raster.profile
    profiles = [raster_profile, base_raster_profile]

    # Unify if the rasters have different grids
    if check_raster_grids(profiles, same_extent=True):
        raster_arr = raster.read()
    else:
        out_rasters = unify_raster_grids(
            base_raster=base_raster, rasters_to_unify=[raster], resampling_method="nearest", same_extent=True
        )
        raster_arr = out_rasters[1][0]

        # Update profiles
        raster_profile = out_rasters[1][1]
        profiles[0] = raster_profile

    # Extract nodata info
    raster_nodata = raster_profile.get("nodata", np.nan)
    base_raster_nodata = base_raster_profile.get("nodata", np.nan)

    # Create mask to apply
    base_raster_arr = base_raster.read(1)
    base_raster_nodata_mask = (base_raster_arr == base_raster_nodata) | np.isnan(base_raster_arr)

    # Apply mask to all bands of input raster
    bands = raster.count
    out_image = np.empty((bands, raster_profile["height"], raster_profile["width"]), dtype=raster_profile["dtype"])
    for i in range(bands):
        band_arr = raster_arr[i]
        band_arr[base_raster_nodata_mask] = raster_nodata
        out_image[i] = band_arr

    out_profile = raster_profile.copy()
    return out_image, out_profile
