from itertools import chain
from numbers import Number
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from rasterio import profiles

from eis_toolkit.exceptions import EmptyDataException, InvalidParameterValueException
from eis_toolkit.utilities.checks.raster import check_raster_profile
from eis_toolkit.utilities.miscellaneous import row_points, toggle_gdal_exceptions
from eis_toolkit.vector_processing.distance_computation import distance_computation


def _check_threshold_criteria_and_value(threshold_criteria, threshold_criteria_value):
    if threshold_criteria in {"lower", "higher"} and not isinstance(threshold_criteria_value, Number):
        raise InvalidParameterValueException(
            f"Expected threshold_criteria_value {threshold_criteria_value} "
            "to be a single number rather than a tuple."
        )
    if threshold_criteria in {"in_between", "outside"}:
        if not isinstance(threshold_criteria_value, tuple):
            raise InvalidParameterValueException(
                f"Expected threshold_criteria_value ({threshold_criteria_value}) to be a tuple rather than a number."
            )
        if threshold_criteria_value[0] >= threshold_criteria_value[1]:
            raise InvalidParameterValueException(
                f"Expected first value in threshold_criteria_value ({threshold_criteria_value})"
                "tuple to be lower than the second."
            )


@beartype
def distance_to_anomaly(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    max_distance: Optional[Number] = None,
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Calculate distance from raster cell to nearest anomaly.

    The criteria for what is anomalous can be defined as a single number and
    criteria text of "higher" or "lower". Alternatively, the definition can be
    a range where values inside (criteria text of "within") or outside are
    marked as anomalous (criteria text of "outside"). If anomaly_raster_profile does
    contain "nodata" key, np.nan is assumed to correspond to nodata values.

    Args:
        anomaly_raster_profile: The raster profile in which the distances
            to the nearest anomalous value are determined.
        anomaly_raster_data: The raster data in which the distances
            to the nearest anomalous value are determined.
        threshold_criteria_value: Value(s) used to define anomalous.
            If the threshold criteria requires a tuple of values,
            the first value should be the minimum and the second
            the maximum value.
        threshold_criteria: Method to define anomalous.
        max_distance: The maximum distance in the output array.

    Returns:
        A 2D numpy array with the distances to anomalies computed
        and the original anomaly raster profile.

    """
    check_raster_profile(raster_profile=anomaly_raster_profile)
    _check_threshold_criteria_and_value(
        threshold_criteria=threshold_criteria, threshold_criteria_value=threshold_criteria_value
    )

    out_image = _distance_to_anomaly(
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria=threshold_criteria,
        threshold_criteria_value=threshold_criteria_value,
        max_distance=max_distance,
    )
    return out_image, anomaly_raster_profile


@beartype
def distance_to_anomaly_gdal(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """Calculate distance from raster cell to nearest anomaly.

    Distance is calculated for each cell in the anomaly raster and saved to a
    new raster at output_path. The criteria for what is anomalous can be
    defined as a single number and criteria text of "higher" or "lower".
    Alternatively, the definition can be a range where values inside
    (criteria text of "within") or outside are marked as anomalous
    (criteria text of "outside"). If anomaly_raster_profile does
    contain "nodata" key, np.nan is assumed to correspond to nodata values.

    Does not work on Windows.

    Args:
        anomaly_raster_profile: The raster profile in which the distances
            to the nearest anomalous value are determined.
        anomaly_raster_data: The raster data in which the distances
            to the nearest anomalous value are determined.
        threshold_criteria_value: Value(s) used to define anomalous.
        threshold_criteria: Method to define anomalous.
        output_path: The path to the raster with the distances to anomalies
            calculated.
        verbose: Whether to print gdal_proximity output.

    Returns:
        The path to the raster with the distances to anomalies calculated.
    """
    check_raster_profile(raster_profile=anomaly_raster_profile)
    _check_threshold_criteria_and_value(
        threshold_criteria=threshold_criteria, threshold_criteria_value=threshold_criteria_value
    )

    return _distance_to_anomaly_gdal(
        output_path=output_path,
        anomaly_raster_profile=anomaly_raster_profile,
        anomaly_raster_data=anomaly_raster_data,
        threshold_criteria=threshold_criteria,
        threshold_criteria_value=threshold_criteria_value,
        verbose=verbose,
    )


def _fits_criteria(
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    anomaly_raster_data: np.ndarray,
    nodata_value: Optional[Number],
) -> np.ndarray:

    criteria_dict = {
        "lower": lambda anomaly_raster_data: anomaly_raster_data < threshold_criteria_value,
        "higher": lambda anomaly_raster_data: anomaly_raster_data > threshold_criteria_value,
        "in_between": lambda anomaly_raster_data: np.logical_and(
            anomaly_raster_data > threshold_criteria_value[0],  # type: ignore
            anomaly_raster_data < threshold_criteria_value[1],  # type: ignore
        ),
        "outside": lambda anomaly_raster_data: np.logical_or(
            anomaly_raster_data < threshold_criteria_value[0],  # type: ignore
            anomaly_raster_data > threshold_criteria_value[1],  # type: ignore
        ),
    }
    mask = anomaly_raster_data == nodata_value if nodata_value is not None else np.isnan(anomaly_raster_data)

    return np.where(mask, False, criteria_dict[threshold_criteria](anomaly_raster_data))


def _write_binary_anomaly_raster(tmp_dir: Path, anomaly_raster_profile, data_fits_criteria: np.ndarray):
    anomaly_raster_binary_path = tmp_dir / "anomaly_raster_binary.tif"

    anomaly_raster_binary_profile = {**anomaly_raster_profile, **dict(dtype=rasterio.uint8, count=1, nodata=None)}
    with rasterio.open(anomaly_raster_binary_path, mode="w", **anomaly_raster_binary_profile) as anomaly_raster_binary:
        anomaly_raster_binary.write(data_fits_criteria.astype(rasterio.uint8), 1)

    return anomaly_raster_binary_path


def _distance_to_anomaly_gdal(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    output_path: Path,
    verbose: bool,
):
    from osgeo_utils import gdal_proximity

    data_fits_criteria = _fits_criteria(
        threshold_criteria=threshold_criteria,
        threshold_criteria_value=threshold_criteria_value,
        anomaly_raster_data=anomaly_raster_data,
        nodata_value=anomaly_raster_profile.get("nodata"),
    )

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        anomaly_raster_binary_path = _write_binary_anomaly_raster(
            tmp_dir=tmp_dir, anomaly_raster_profile=anomaly_raster_profile, data_fits_criteria=data_fits_criteria
        )
        with toggle_gdal_exceptions():
            gdal_proximity.gdal_proximity(
                src_filename=str(anomaly_raster_binary_path),
                dst_filename=str(output_path),
                alg_options=("VALUES=1", "DISTUNITS=GEO"),
                quiet=not verbose,
            )

    return output_path


def _distance_to_anomaly(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    max_distance: Optional[Number],
) -> np.ndarray:
    data_fits_criteria = _fits_criteria(
        threshold_criteria=threshold_criteria,
        threshold_criteria_value=threshold_criteria_value,
        anomaly_raster_data=anomaly_raster_data,
        nodata_value=anomaly_raster_profile.get("nodata"),
    )
    if np.sum(data_fits_criteria) == 0:
        raise EmptyDataException(
            " ".join(
                [
                    "Expected the passed threshold criteria to match at least some data.",
                    f"Check that the values of threshold_criteria ({threshold_criteria})",
                    f"and threshold_criteria_value {threshold_criteria_value}",
                    "match at least part of the data.",
                ]
            )
        )

    cols = np.arange(anomaly_raster_data.shape[1])
    rows = np.arange(anomaly_raster_data.shape[0])

    all_points_by_rows = [
        row_points(row=row, cols=cols[data_fits_criteria[row]], raster_transform=anomaly_raster_profile["transform"])
        for row in rows
    ]
    all_points = list(chain(*all_points_by_rows))
    all_points_gdf = gpd.GeoDataFrame(geometry=all_points, crs=anomaly_raster_profile["crs"])

    distance_array = distance_computation(
        raster_profile=anomaly_raster_profile, geodataframe=all_points_gdf, max_distance=max_distance
    )

    return distance_array
