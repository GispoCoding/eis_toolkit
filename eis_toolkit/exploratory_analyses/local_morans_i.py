import numpy as np
from esda.moran import Moran_Local
import libpysal
import geopandas as gpd
from beartype import beartype
from beartype.typing import Literal

from eis_toolkit import exceptions

@beartype
def _local_morans_i(
    gdf: gpd.GeoDataFrame,
    column: str,
    weight_type: Literal["queen", "knn"],
    k: int,
    permutations: int
) -> gpd.GeoDataFrame:

    gdf['lon'] = gdf['geometry'].x
    gdf['lat'] = gdf['geometry'].y

    if weight_type == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf)
    elif weight_type == "knn":
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k)
    else:
        raise ValueError("Invalid weight_type. Use 'queen' or 'knn'.")

    if len(gdf[column]) != len(w.weights):
        raise ValueError('Dimension mismatch between data and weights matrix.')
    
    moran_loc = Moran_Local(gdf[column], w, permutations=permutations)
    
    gdf[f"{column}_local_moran_I"] = moran_loc.Is
    gdf[f"{column}_p_value"] = moran_loc.p_sim

    gdf[f"{column}_p_value"].fillna(value=np.nan, inplace=True)

    return gdf

@beartype
def local_morans_i(
    gdf: gpd.GeoDataFrame,
    column: str,
    weight_type: Literal["queen", "knn"] = "queen",
    k: int = 2,
    permutations: int = 999
) -> gpd.GeoDataFrame:
    """Do a local morans I calculation for the area.

    Args:
        area: The geodataframe that contains the area to be examined with local morans I.

    Returns:
        Geodataframe containing the calculations.

    Raises:
        EmptyDataFrameException if input geodataframe is empty.
    """
    if gdf.shape[0] == 0:
        raise exceptions.EmptyDataFrameException("Geodataframe is empty.")

    calculations = _local_morans_i(gdf, column, weight_type, k, permutations)

    return calculations
