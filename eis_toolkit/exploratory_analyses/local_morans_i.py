import geopandas as gpd
import libpysal
import numpy as np
from beartype import beartype
from beartype.typing import Literal
from esda.moran import Moran_Local

from eis_toolkit import exceptions
from eis_toolkit.exceptions import InvalidParameterValueException


@beartype
def _local_morans_i(
    gdf: gpd.GeoDataFrame, column: str, weight_type: Literal["queen", "knn"], k: int, permutations: int
) -> gpd.GeoDataFrame:

    if weight_type == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf)
    elif weight_type == "knn":
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k)
    else:
        raise InvalidParameterValueException("Invalid weight_type. Use 'queen' or 'knn'.")

    w.transform = "R"

    if len(gdf[column]) != len(w.weights):
        raise InvalidParameterValueException("Dimension mismatch between data and weights matrix.")

    moran_loc = Moran_Local(gdf[column], w, permutations=permutations)

    gdf[f"{column}_local_moran_I"] = moran_loc.Is
    gdf[f"{column}_local_moran_I_p_value"] = moran_loc.p_sim

    gdf[f"{column}_local_moran_I_p_value"].fillna(value=np.nan, inplace=True)

    return gdf


@beartype
def local_morans_i(
    gdf: gpd.GeoDataFrame,
    column: str,
    weight_type: Literal["queen", "knn"] = "queen",
    k: int = 4,
    permutations: int = 999,
) -> gpd.GeoDataFrame:
    """Execute Local Moran's I calculation for the data.

    Args:
        gdf: The geodataframe that contains the data to be examined with local morans I.
        column: The column to be used in the analysis.
        weight_type: The type of spatial weights matrix to be used. Defaults to "queen".
        k: Number of nearest neighbors for the KNN weights matrix. Defaults to 4.
        permutations: Number of permutations for significance testing. Defaults to 999.

    Returns:
        Geodataframe appended with two new columns: one with Local Moran's I
          statistic and one with p-value for the statistic.

    Raises:
        EmptyDataFrameException: The input geodataframe is empty.
    """
    if gdf.shape[0] == 0:
        raise exceptions.EmptyDataFrameException("Geodataframe is empty.")

    if column not in gdf.columns:
        raise exceptions.InvalidParameterValueException(f"Column '{column}' not found in the GeoDataFrame.")

    if k < 1:
        raise exceptions.InvalidParameterValueException("k must be > 0.")

    if permutations < 100:
        raise exceptions.InvalidParameterValueException("permutations must be > 99.")

    calculations = _local_morans_i(gdf, column, weight_type, k, permutations)

    return calculations
