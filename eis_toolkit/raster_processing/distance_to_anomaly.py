from itertools import chain
from numbers import Number
from os import path
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, Union
from rasterio import profiles

from eis_toolkit.exceptions import EmptyDataException, InvalidParameterValueException, NumericValueSignException
from eis_toolkit.utilities.checks.raster import check_raster_profile
from eis_toolkit.utilities.miscellaneous import row_points
from eis_toolkit.vector_processing._distance_computation_numba import distance_computation_numba


@beartype
def distance_to_anomaly(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    max_distance: Optional[Number] = None,
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Calculate distance from raster cell to nearest anomaly.

    If `osgeo_utils.gdal_proximity` is not available, will print a warning and fallback to a slower implementation.

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
    import typer

    try:
        from osgeo_utils import gdal_proximity  # noqa: F401

        func = _distance_to_anomaly_gdal
    except ModuleNotFoundError:
        func = _distance_to_anomaly_numba
        typer.echo("WARNING: Falling back to a slower implementation as 'gdal_proximity' was not available!")

    check_raster_profile(raster_profile=anomaly_raster_profile)
    _check_threshold_criteria_and_value(
        threshold_criteria=threshold_criteria, threshold_criteria_value=threshold_criteria_value
    )
    if max_distance is not None and max_distance <= 0:
        raise NumericValueSignException("Expected max distance to be a positive number.")

    distance_raster_data = anomaly_raster_data.copy()
    distance_raster_data.astype(np.float64)
    out_profile = anomaly_raster_profile.copy()
    out_profile["dtype"] = np.float32
    out_profile["count"] = 1

    # We need to convert nodata to a negative number in case it is non-negative in the input raster
    if out_profile["nodata"] >= 0.0:
        old_nodata = out_profile["nodata"]
        new_nodata = -9999
        distance_raster_data = np.where(np.isin(distance_raster_data, old_nodata), new_nodata, distance_raster_data)
        out_profile["nodata"] = new_nodata

    out_array, out_profile = func(
        anomaly_raster_profile=out_profile,
        anomaly_raster_data=distance_raster_data,
        threshold_criteria=threshold_criteria,
        threshold_criteria_value=threshold_criteria_value,
        max_distance=max_distance,
    )
    return out_array, out_profile


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


def _validate_threshold_criteria(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
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
    return data_fits_criteria


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
    max_distance: Optional[Number],
    # output_path: Path,
    verbose: bool = False,
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:

    from osgeo_utils import gdal_proximity

    from eis_toolkit.utilities.gdal import toggle_gdal_exceptions

    data_fits_criteria = _validate_threshold_criteria(
        anomaly_raster_profile,
        anomaly_raster_data,
        threshold_criteria_value,
        threshold_criteria,
    )

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        anomaly_raster_binary_path = _write_binary_anomaly_raster(
            tmp_dir=tmp_dir, anomaly_raster_profile=anomaly_raster_profile, data_fits_criteria=data_fits_criteria
        )
        tmp_out_raster = path.join(tmp_dir, "temp_output.tif")
        with toggle_gdal_exceptions():
            gdal_proximity.gdal_proximity(
                src_filename=str(anomaly_raster_binary_path),
                dst_filename=tmp_out_raster,
                alg_options=("VALUES=1", "DISTUNITS=GEO"),
                quiet=not verbose,
            )
            with rasterio.open(tmp_out_raster) as out_raster:
                out_array = out_raster.read(1)

    if max_distance is not None:
        out_array[out_array > max_distance] = max_distance

    return out_array, anomaly_raster_profile


def _distance_to_anomaly_numba(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    max_distance: Optional[Number],
) -> Tuple[np.ndarray, profiles.Profile]:
    data_fits_criteria = _validate_threshold_criteria(
        anomaly_raster_profile,
        anomaly_raster_data,
        threshold_criteria_value,
        threshold_criteria,
    )

    cols = np.arange(anomaly_raster_data.shape[1])
    rows = np.arange(anomaly_raster_data.shape[0])

    all_points_by_rows = [
        row_points(row=row, cols=cols[data_fits_criteria[row]], raster_transform=anomaly_raster_profile["transform"])
        for row in rows
    ]
    all_points = list(chain(*all_points_by_rows))
    all_points_gdf = gpd.GeoDataFrame(geometry=all_points, crs=anomaly_raster_profile["crs"])

    distance_array = distance_computation_numba(
        raster_profile=anomaly_raster_profile, geodataframe=all_points_gdf, max_distance=max_distance
    )
    out_meta = anomaly_raster_profile.copy()

    # Update metadata
    out_meta["dtype"] = distance_array.dtype.name
    out_meta["count"] = 1

    return distance_array, out_meta


# def _distance_to_anomaly_gdal_alt(
#     anomaly_raster_profile: Union[profiles.Profile, dict],
#     anomaly_raster_data: np.ndarray,
#     threshold_criteria_value: Union[Tuple[Number, Number], Number],
#     threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
#     max_distance: Optional[Number],
# ) -> Tuple[np.ndarray, profiles.Profile]:

#     from osgeo import gdal
#     from eis_toolkit.utilities.gdal import toggle_gdal_exceptions

#     with toggle_gdal_exceptions():
#         data_fits_criteria = _validate_threshold_criteria(
#             anomaly_raster_profile,
#             anomaly_raster_data,
#             threshold_criteria_value,
#             threshold_criteria,
#         )
#         # converting True False values to binary formant.
#         converted_values = np.where(data_fits_criteria, 1, 0)
#         driver = gdal.GetDriverByName("MEM")

#         width = anomaly_raster_profile["width"]
#         height = anomaly_raster_profile["height"]
#         temp_raster = driver.Create("", width, height, 1, gdal.GDT_Float32)
#         transformation = anomaly_raster_profile["transform"]
#         x_geo = (transformation.c, transformation.a, transformation.b)
#         y_geo = (transformation.f, transformation.d, transformation.e)
#         temp_raster.SetGeoTransform(x_geo + y_geo)
#         crs = anomaly_raster_profile["crs"].to_wkt()
#         band = temp_raster.GetRasterBand(1)
#         band.WriteArray(converted_values)
#         nodatavalue = anomaly_raster_profile["nodata"]
#         band.SetNoDataValue(nodatavalue)

#         # Create empty proximity raster
#         out_raster = driver.Create("", width, height, 1, gdal.GDT_Float32)
#         out_raster.SetGeoTransform(x_geo + y_geo)
#         out_raster.SetProjection(crs)
#         out_band = out_raster.GetRasterBand(1)
#         out_band.SetNoDataValue(nodatavalue)
#         options = ["values=1", "distunits=GEO"]

#         # Compute proximity
#         gdal.ComputeProximity(band, out_band, options)

#         # Create outputs
#         out_array = out_band.ReadAsArray()
#         if max_distance is not None:
#             out_array[out_array > max_distance] = max_distance
#         out_meta = anomaly_raster_profile.copy()

#         # Update metadata
#         out_meta["dtype"] = out_array.dtype.name
#         out_meta["count"] = 1

#     return out_array, out_meta
