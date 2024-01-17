import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from rasterio import features, profiles, transform
from rasterio.enums import MergeAlg

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidColumnException,
    InvalidParameterValueException,
    NumericValueSignException,
)


@beartype
def rasterize_vector(
    geodataframe: gpd.GeoDataFrame,
    resolution: Optional[float] = None,
    value_column: Optional[str] = None,
    default_value: float = 1.0,
    fill_value: float = 0.0,
    base_raster_profile: Optional[Union[profiles.Profile, dict]] = None,
    buffer_value: Optional[float] = None,
    merge_strategy: Literal["replace", "add"] = "replace",
) -> Tuple[np.ndarray, dict]:
    """Transform vector data into raster data.

    Args:
        geodataframe: The vector dataframe to be rasterized.
        resolution: The resolution i.e. cell size of the output raster.
            Optional if base_raster_profile is given.
        value_column: The column name with values for each geometry.
            If None, then default_value is used for all geometries.
        default_value: Default value burned into raster cells based on geometries.
        base_raster_profile: Base raster profile
            to be used for determining the grid on which vectors are
            burned in. If None, the geometries and provided resolution
            value are used to compute grid.
        fill_value: Value used outside the burned/rasterized geometry cells.
        buffer_value: For adding a buffer around passed
            geometries before rasterization.
        merge_strategy: How to handle overlapping geometries.
            "add" causes overlapping geometries to add together the
            values while "replace" does not. Adding them together is the
            basis for density computations where the density can be
            calculated by using a default value of 1.0 and the sum in
            each cell is the count of intersecting geometries.

    Returns:
        Rasterized vector data and metadata.

    Raises:
        EmptyDataFrameException: The geodataframe does not contain geometries.
        InvalidColumnException: Given value_column is not in the input geodataframe.
        InvalidParameterValueException: No resolution or base_raster_profile is given,
            or base_raster_profile has the wrong type.
        NumericValueSignException: Input resolution value is zero or negative, or input
            buffer_value is negative.
    """

    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if resolution is None and base_raster_profile is None:
        raise InvalidParameterValueException("Expected either resolution or base_raster_profile to be given.")

    if resolution is not None and resolution <= 0:
        raise NumericValueSignException(f"Expected a positive resolution value ({dict(resolution=resolution)})")

    if value_column is not None and value_column not in geodataframe.columns:
        raise InvalidColumnException(f"Expected value_column ({value_column}) to be contained in geodataframe columns.")

    if buffer_value is not None and buffer_value < 0:
        raise NumericValueSignException(f"Expected a positive buffer_value ({dict(buffer_value=buffer_value)})")

    if base_raster_profile is not None and not isinstance(base_raster_profile, (profiles.Profile, dict)):
        raise InvalidParameterValueException(
            f"Expected base_raster_profile ({type(base_raster_profile)}) to be dict or rasterio.profiles.Profile."
        )

    if buffer_value is not None:
        geodataframe = geodataframe.copy()
        geodataframe["geometry"] = geodataframe["geometry"].apply(lambda geom: geom.buffer(buffer_value))

    return _rasterize_vector(
        geodataframe=geodataframe,
        value_column=value_column,
        default_value=default_value,
        fill_value=fill_value,
        base_raster_profile=base_raster_profile,
        resolution=resolution,
        merge_alg=getattr(MergeAlg, merge_strategy),
    )


def _transform_from_geometries(
    geodataframe: gpd.GeoDataFrame, resolution: float
) -> Tuple[float, float, transform.Affine]:
    """Determine transform from the input geometries.

    Returns:
        Width, height and transform of the raster in a tuple.
    """
    min_x, min_y, max_x, max_y = geodataframe.total_bounds
    width = (max_x - min_x) / resolution
    height = (max_y - min_y) / resolution

    out_transform = transform.from_bounds(min_x, min_y, max_x, max_y, width=width, height=height)
    return width, height, out_transform


def _rasterize_vector(
    geodataframe: gpd.GeoDataFrame,
    value_column: Optional[str],
    default_value: float,
    fill_value: float,
    base_raster_profile: Optional[Union[profiles.Profile, dict]],
    resolution: Optional[float],
    merge_alg: MergeAlg,
) -> Tuple[np.ndarray, dict]:
    # rasterio.features.rasterize expects a shapes parameter which is
    # an iterable of tuples where the first value is a geometry and
    # the other a value for the geometry
    # Alternatively, if there are not values for each geometry,
    # an iterable of geometries can be passed
    geometries = geodataframe["geometry"].values
    values = geodataframe[value_column].values if value_column is not None else None
    geometry_value_pairs = list(geometries) if values is None else list(zip(geometries, values))

    if base_raster_profile is None and resolution is not None:
        width, height, out_transform = _transform_from_geometries(geodataframe=geodataframe, resolution=resolution)
    elif base_raster_profile is not None:
        width, height, out_transform = (
            base_raster_profile["width"],
            base_raster_profile["height"],
            base_raster_profile["transform"],
        )
    else:
        raise InvalidParameterValueException("Expected resolution or base_raster_profile to be given.")

    out_raster_array = features.rasterize(
        shapes=geometry_value_pairs,
        # fill and default_value can be floats even though typing claims otherwise
        fill=fill_value,
        default_value=default_value,
        transform=out_transform,
        out_shape=(round(height), round(width)),
        merge_alg=merge_alg,
    )
    return out_raster_array, dict(transform=out_transform, height=height, width=width)
