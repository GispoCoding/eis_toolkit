import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from rasterio import profiles

from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector


@beartype
def vector_density(
    geodataframe: gpd.GeoDataFrame,
    resolution: Optional[float] = None,
    base_raster_profile: Optional[Union[profiles.Profile, dict]] = None,
    buffer_value: Optional[float] = None,
    statistic: Literal["density", "count"] = "density",
) -> Tuple[np.ndarray, dict]:
    """Compute density of geometries within raster.

    Args:
        geodataframe: The dataframe with vectors
            of which density is computed.
        resolution: The resolution i.e. cell size of the output raster.
            Optional if base_raster_profile is given.
        base_raster_profile: Base raster profile
            to be used for determining the grid on which vectors are
            burned in. If None, the geometries and provided resolution
            value are used to compute grid.
        buffer_value: For adding a buffer around passed
            geometries before computing density.

    Returns:
        Computed density of vector data and metadata.
    """
    out_raster_array, out_metadata = rasterize_vector(
        geodataframe=geodataframe,
        resolution=resolution,
        base_raster_profile=base_raster_profile,
        buffer_value=buffer_value,
        value_column=None,
        default_value=1.0,
        fill_value=0.0,
        merge_strategy="add",
    )
    max_count = np.max(out_raster_array)
    if statistic == "count" or np.isclose(max_count, 0.0):
        return out_raster_array, out_metadata
    else:
        return (out_raster_array / max_count), out_metadata
