from typing import List

import geopandas
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.geometry import check_geometry_types
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def get_tpr_poa(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    deposits: geopandas.GeoDataFrame,
    band: int = 1,
) -> pd.DataFrame:
    """Calculate true positive rate and proportion of area values.

    Function calculates true positive rate and proportion of area values for different thresholds which are
    determined from inputted deposit locations and mineral prospectivity map.

    Args:
        raster (rasterio.io.DatasetReader): Mineral prospectivity map.
        deposits (geopandas.GeoDataFrame): Mineral deposit locations.
        band (int): band index of the mineral prospectivity map, defaults to 1.

    Returns:
        data_frame: data frame containing true positive rate, proportion of area and threshold values.

    Raises:
        NonMatchingCrsException: The raster and polygons are not in the same crs.
        NotApplicableGeometryTypeException: The input geometries contain non-polygon features.
    """
    geometries = deposits["geometry"]

    if not check_matching_crs(
        objects=[raster, geometries],
    ):
        raise NonMatchingCrsException

    if not check_geometry_types(
        geometries=geometries,
        allowed_types=["Point"],
    ):
        raise NotApplicableGeometryTypeException

    data_array = raster.read(band)
    data_array = (data_array - data_array.min()) / (data_array.max() - data_array.min())  # rescale to 0-1

    # Select only deposits that are whitin raster bounds
    deposits = deposits.cx[raster.bounds.left : raster.bounds.right, raster.bounds.bottom : raster.bounds.top]

    rows, cols = rasterio.transform.rowcol(raster.transform, deposits.geometry.x, deposits.geometry.y)
    deposit_scores = data_array[rows, cols]
    thresholds = np.flip(np.unique(deposit_scores))

    true_positive_rate_values = []
    proportion_of_area_values = []

    for threshold in thresholds:
        true_positive_rate_values.append((deposit_scores >= threshold).sum() / deposit_scores.size)
        proportion_of_area_values.append((data_array >= threshold).sum() / data_array.size)

    data_frame = pd.DataFrame(
        {
            "true_positive_rate": true_positive_rate_values,
            "proportion_of_area": proportion_of_area_values,
            "threshold": thresholds,
        }
    )

    return data_frame


def plot_rate_curve(  # type: ignore[no-any-unimported]
    true_positive_rate_values: List,
    proportion_of_area_values: List,
) -> matplotlib.figure.Figure:
    """Plot success or prediction rate curve.

    Type of plot depends on the given deposits. If deposits were used for model training, then the plot is known as
    success rate curve. If deposits were not used for model training then the plot is known as prediction rate plot. In
    both cases x-axis indicates the proportion of area that is considired to be prospective and y-axis indicates true
    positive rate. See (?add reference).

    Args:
        true_positive_rate_values (List): True positive rate values, y-coordinates of the plot.
        proportion_of_area_values (List): Proportion of area values, x-coordinates of the plot.

    Returns:
        matplotlib.figure.Figure: Success or prediction rate curve plot.
    """

    fig = plt.figure(figsize=(10, 7))
    plt.plot(proportion_of_area_values, true_positive_rate_values)

    return fig


rast = rasterio.open("tests/data/remote/small_raster.tif")

df = pd.DataFrame(
    {
        "x": [384829, 384803, 384807, 384793, 384773, 384785],
        "y": [6671311, 6671295, 6671277, 6671293, 6671343, 6671357],
    }
)

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y, crs="EPSG:3067"))
coords = get_tpr_poa(rast, gdf)
plot_rate_curve(coords.true_positive_rate, coords.proportion_of_area)
