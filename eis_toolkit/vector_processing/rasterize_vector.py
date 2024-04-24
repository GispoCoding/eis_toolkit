import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Union
from rasterio import features, profiles, transform
from rasterio.enums import MergeAlg

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidColumnException,
    NonMatchingCrsException,
    NumericValueSignException,
)
from eis_toolkit.utilities.checks.raster import check_raster_profile


@beartype
def rasterize_vector(
    geodataframe: gpd.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    value_column: Optional[str] = None,
    default_value: float = 1.0,
    fill_value: float = 0.0,
    buffer_value: Optional[float] = None,
    merge_strategy: Literal["replace", "add"] = "replace",
) -> np.ndarray:
    """Transform vector data into raster data.

    Args:
        geodataframe: The vector dataframe to be rasterized.
        raster_profile: The raster profile used for output grid properties.
            Needs to include at least CRS, transform, width and height.
        value_column: The column name with values for each geometry.
            If None, then default_value is used for all geometries.
        default_value: Default value burned into raster cells based on geometries.
        fill_value: Value used outside the burned/rasterized geometry cells.
        buffer_value: For adding a buffer around passed geometries before rasterization.
        merge_strategy: How to handle overlapping geometries.
            "add" causes overlapping geometries to add together the
            values while "replace" does not. Adding them together is the
            basis for density computations where the density can be
            calculated by using a default value of 1.0 and the sum in
            each cell is the count of intersecting geometries.

    Returns:
        Rasterized vector data..

    Raises:
        EmptyDataFrameException: The geodataframe does not contain geometries.
        InvalidColumnException: Given value_column is not in the input geodataframe.
        NonMatchingCrsException: The input GeoDataFrame and raster profile have mismatching CRS.
        NumericValueSignException: Input resolution value is zero or negative, or input
            buffer_value is negative.
    """
    if geodataframe.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")
    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    if value_column is not None and value_column not in geodataframe.columns:
        raise InvalidColumnException(f"Expected value_column ({value_column}) to be contained in geodataframe columns.")
    check_raster_profile(raster_profile)

    if buffer_value is not None:
        if buffer_value < 0:
            raise NumericValueSignException(f"Expected a positive buffer_value ({dict(buffer_value=buffer_value)})")

        geodataframe = geodataframe.copy()
        geodataframe["geometry"] = geodataframe["geometry"].apply(lambda geom: geom.buffer(buffer_value))

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    out_image = _rasterize_vector(
        geodataframe=geodataframe,
        raster_width=raster_width,
        raster_height=raster_height,
        raster_transform=raster_transform,
        value_column=value_column,
        default_value=default_value,
        fill_value=fill_value,
        merge_alg=getattr(MergeAlg, merge_strategy),
    )
    return out_image


def _rasterize_vector(
    geodataframe: gpd.GeoDataFrame,
    raster_width: int,
    raster_height: int,
    raster_transform: transform.Affine,
    value_column: Optional[str],
    default_value: float,
    fill_value: float,
    merge_alg: MergeAlg,
) -> np.ndarray:
    # rasterio.features.rasterize expects a shapes parameter which is
    # an iterable of tuples where the first value is a geometry and
    # the other a value for the geometry
    # Alternatively, if there are not values for each geometry,
    # an iterable of geometries can be passed

    geometries = geodataframe["geometry"].values
    values = geodataframe[value_column].values if value_column is not None else None
    geometry_value_pairs = list(geometries) if values is None else list(zip(geometries, values))

    out_raster_array = features.rasterize(
        shapes=geometry_value_pairs,
        # fill and default_value can be floats even though typing claims otherwise
        fill=fill_value,
        default_value=default_value,
        transform=raster_transform,
        out_shape=(raster_height, raster_width),
        merge_alg=merge_alg,
    )
    return out_raster_array
