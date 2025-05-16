import geopandas as gpd
import numpy as np
import pytest

from eis_toolkit import exceptions
from eis_toolkit.exploratory_analyses.local_morans_i import local_morans_i


def test_local_morans_i_queen_correctness():
    """Test Local Moran's I Queen correctness."""

    permutations = 999

    column = "gdp_md_est"
    data = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf = gpd.GeoDataFrame(data)
    gdf20 = gdf.head(20)

    Is = [
        -0.0,
        0.07475982388811646,
        -0.0,
        0.32707308155951376,
        0.3270730815595137,
        0.02567026668185298,
        0.06875988330128255,
        0.014784750951796788,
        0.014784750951796785,
        0.04431219450688388,
        0.04431219450688389,
        0.07606167537417435,
        0.07674902657188765,
        0.07510349948697309,
        0.08027860346365057,
        0.08027860346365055,
        0.0766008275457467,
        0.0766008275457467,
        -0.017419349937576573,
        -0.0,
    ]

    p_sims = [
        0.001,
        0.365,
        0.001,
        0.056,
        0.056,
        0.191,
        0.355,
        0.206,
        0.38,
        0.201,
        0.256,
        0.464,
        0.351,
        0.24,
        0.224,
        0.386,
        0.397,
        0.338,
        0.299,
        0.001,
    ]

    result = local_morans_i(gdf=gdf20, column=column, weight_type="queen", permutations=permutations)

    np.testing.assert_allclose(result[f"{column}_local_moran_I"], Is, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(result[f"{column}_local_moran_I_p_value"], p_sims, rtol=0.1, atol=0.1)


def test_local_morans_i_knn_correctness():
    """Test Local Moran's I KNN correctness."""

    k = 4
    permutations = 999

    column = "gdp_md_est"
    data = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf = gpd.GeoDataFrame(data)
    gdf20 = gdf.head(20)

    Is = [
        0.03686709680905326,
        0.07635702591252187,
        0.07990131850915279,
        0.06552029316560676,
        -0.8031053484182811,
        0.048272793989863144,
        0.051515005797283464,
        0.03846954477931574,
        0.010137393086155687,
        0.051762074257733624,
        0.05895594777225281,
        0.0768224164382028,
        0.07889650044641662,
        0.07492029251731681,
        0.07855252515119235,
        0.07851482805880286,
        -0.26650904930879504,
        -0.25076447340691294,
        -0.015081612933344679,
        -0.2666687928014803,
    ]

    p_sims = [
        0.223,
        0.094,
        0.082,
        0.148,
        0.257,
        0.465,
        0.396,
        0.27,
        0.469,
        0.319,
        0.319,
        0.115,
        0.126,
        0.095,
        0.087,
        0.133,
        0.035,
        0.054,
        0.416,
        0.033,
    ]

    result = local_morans_i(gdf20, column, "knn", k=k, permutations=permutations)

    np.testing.assert_allclose(result[f"{column}_local_moran_I"], Is, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(result[f"{column}_local_moran_I_p_value"], p_sims, rtol=0.1, atol=0.1)


def test_empty_geodataframe():
    """Test Local Moran's I raises EmptyDataFrameException."""

    empty_gdf = gpd.GeoDataFrame()

    # Use pytest.raises to check the expected exception
    with pytest.raises(exceptions.EmptyDataFrameException):
        local_morans_i(empty_gdf, column="value", weight_type="queen", k=2, permutations=999)


def test_geodataframe_missing_column():
    """Test Local Moran's I raises InvalidParameterValueException for missing column."""

    gdf = gpd.GeoDataFrame({"test_col": [1, 2, 3]})

    with pytest.raises(exceptions.InvalidColumnException):
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
