import geopandas
import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.geometry import check_geometry_types
from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException


def _calculate_base_metrics(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    deposits: geopandas.GeoDataFrame,
    band: int = 1,
    negatives: geopandas.GeoDataFrame = None,
) -> pd.DataFrame:
    data_array = raster.read(band)

    # Select only deposits that are within raster bounds
    deposits = deposits.cx[
        raster.bounds.left : raster.bounds.right,  # noqa: E203
        raster.bounds.bottom : raster.bounds.top,  # noqa: E203
    ]

    deposit_rows, deposit_cols = rasterio.transform.rowcol(raster.transform, deposits.geometry.x, deposits.geometry.y)
    deposit_scores = data_array[deposit_rows, deposit_cols]
    threshold_values = np.flip(np.unique(deposit_scores))

    if threshold_values.max() < data_array.max():
        threshold_values = np.concatenate(([data_array.max()], threshold_values))

    if threshold_values.min() > data_array.min():
        threshold_values = np.concatenate((threshold_values, [data_array.min()]))

    true_positive_rate_values = []
    proportion_of_area_values_values = []

    for threshold_value in threshold_values:
        true_positive_rate_values.append((deposit_scores >= threshold_value).sum() / deposit_scores.size)
        proportion_of_area_values_values.append((data_array >= threshold_value).sum() / data_array.size)

    base_metrics = pd.DataFrame(
        {
            "true_positive_rate_values": true_positive_rate_values,
            "proportion_of_area_values": proportion_of_area_values_values,
            "threshold_values": threshold_values,
        }
    )

    if negatives is not None:
        # Select only negatives that are within raster bounds
        negatives = negatives.cx[
            raster.bounds.left : raster.bounds.right,  # noqa: E203
            raster.bounds.bottom : raster.bounds.top,  # noqa: E203
        ]
        negatives_rows, negatives_cols = rasterio.transform.rowcol(
            raster.transform, negatives.geometry.x, negatives.geometry.y
        )
        negatives_scores = data_array[negatives_rows, negatives_cols]

        false_positive_rate_values = []
        for threshold_values in threshold_values:
            false_positive_rate_values.append((negatives_scores >= threshold_values).sum() / negatives_scores.size)

        base_metrics["false_positive_rate_values"] = false_positive_rate_values

    return base_metrics


def calculate_base_metrics(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    deposits: geopandas.GeoDataFrame,
    band: int = 1,
    negatives: geopandas.GeoDataFrame = None,
) -> pd.DataFrame:
    """Calculate true positive rate, proportion of area and false positive rate values for different thresholds.

    Function calculates true positive rate, proportion of area and false positive rate values for different thresholds
    which are determined from inputted deposit locations and mineral prospectivity map. Note that calculation of false
    positive rate is optional and is only done if negative point locations are provided.

    Args:
        raster (rasterio.io.DatasetReader): Mineral prospectivity map or evidence layer.
        deposits (geopandas.GeoDataFrame): Mineral deposit locations as points.
        band (int): band index of the mineral prospectivity map, defaults to 1.
        negatives (gepandas.GeoDataFrame): Negative locations as points.

    Returns:
        base_metrics: data frame containing true positive rate, proportion of area, threshold_values and false positive
            rate (optional) values.

    Raises:
        NonMatchingCrsException: The raster and point data are not in the same crs.
        NotApplicableGeometryTypeException: The input geometries contain non-point features.
    """
    if negatives is not None:
        geometries = pd.concat([deposits, negatives]).geometry
    else:
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

    base_metrics = _calculate_base_metrics(raster=raster, deposits=deposits, band=band, negatives=negatives)

    return base_metrics
