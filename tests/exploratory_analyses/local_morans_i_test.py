import geopandas as gpd
import libpysal
import numpy as np
import pytest
from esda.moran import Moran_Local

from eis_toolkit import exceptions
from eis_toolkit.exploratory_analyses.local_morans_i import local_morans_i


def test_local_morans_i_queen_correctness():
    """Test Local Moran's I Queen correctness."""

    permutations = 999

    column = "gdp_md_est"
    data = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf = gpd.GeoDataFrame(data)

    w = libpysal.weights.Queen.from_dataframe(gdf)

    moran_loc = Moran_Local(gdf[column], w, permutations=permutations)

    result = local_morans_i(gdf=gdf, column=column, weight_type="queen", permutations=permutations)

    np.testing.assert_allclose(result[f"{column}_local_moran_I"], moran_loc.Is, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(result[f"{column}_local_moran_I_p_value"], moran_loc.p_sim, rtol=0.1, atol=0.1)


def test_local_morans_i_knn_correctness():
    """Test Local Moran's I KNN correctness."""

    k = 4
    permutations = 999

    column = "gdp_md_est"
    data = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf = gpd.GeoDataFrame(data)

    w = libpysal.weights.KNN.from_dataframe(gdf, k=k)
    moran_loc = Moran_Local(gdf[column], w, permutations=permutations)

    result = local_morans_i(gdf, column, "knn", k=k, permutations=permutations)

    np.testing.assert_allclose(result[f"{column}_local_moran_I"], moran_loc.Is, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(result[f"{column}_local_moran_I_p_value"], moran_loc.p_sim, rtol=0.1, atol=0.1)


def test_empty_geodataframe():
    """Test Local Moran's I raises EmptyDataFrameException."""

    empty_gdf = gpd.GeoDataFrame()

    # Use pytest.raises to check the expected exception
    with pytest.raises(exceptions.EmptyDataFrameException):
        local_morans_i(empty_gdf, column="value", weight_type="queen", k=2, permutations=999)


def test_geodataframe_missing_column():
    """Test Local Moran's I raises InvalidParameterValueException for missing column."""

    gdf = gpd.GeoDataFrame({"test_col": [1, 2, 3]})

    with pytest.raises(exceptions.InvalidParameterValueException):
        local_morans_i(gdf, column="value", weight_type="queen", k=4, permutations=999)


def test_invalid_k_value():
    """Test Local Moran's I raises InvalidParameterValueException for k value under 1."""

    gdf = gpd.GeoDataFrame({"value": [1, 2, 3]})

    with pytest.raises(exceptions.InvalidParameterValueException):
        local_morans_i(gdf, column="value", weight_type="queen", k=0, permutations=999)


def test_invalid_permutations_value():
    """Test Local Moran's I raises InvalidParameterValueException for permutations value under 100."""

    gdf = gpd.GeoDataFrame({"value": [1, 2, 3]})

    with pytest.raises(exceptions.InvalidParameterValueException):
        local_morans_i(gdf, column="value", weight_type="queen", k=4, permutations=99)
