import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils
from eis_toolkit.checks import parameter


# Core functions
def _z_score_normalization_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    with_mean: bool = True,
    with_sd: bool = True,
    nodata_value: Optional[int | float] = None,
) -> Tuple[np.ndarray, float, float]:

    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan

    mean = 0 if not with_mean else np.nanmean(out_array).astype(np.float32)
    std = 1 if not with_sd else np.nanstd(out_array).astype(np.float32)
    out_array = (out_array - mean) / std

    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array, mean, std


def _minmax_scaling_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    new_range: Tuple[int | float, int | float] = (0, 1),
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:

    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan

    min = np.nanmin(out_array)
    max = np.nanmax(out_array)
    scaled_min = new_range[0]
    scaled_max = new_range[1]

    scaler = (out_array - min) / (max - min)
    out_array = (scaler * (scaled_max - scaled_min)) + scaled_min

    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def _z_score_normalization_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    with_mean: List[bool] = [True],
    with_sd: List[bool] = [True],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
    raster = in_data

    if not bands:
        bands = list(range(1, raster.count + 1))

    expanded_args = utils.expand_args(selection=bands, nodata=nodata, with_mean=with_mean, with_sd=with_sd)
    nodata = expanded_args["nodata"]
    with_mean = expanded_args["with_mean"]
    with_sd = expanded_args["with_sd"]

    out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
    out_settings = {}

    for i, band_idx in enumerate(bands_idx):
        nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]

        out_array[band_idx], out_mean, out_std = _z_score_normalization_core(
            data_array=out_array[band_idx], with_mean=with_mean[i], with_sd=with_sd[i], nodata_value=nodata_value
        )

        current_transform = f"band {band_idx + 1}"
        current_settings = {
            "band_origin": bands[i],
            "mean": out_mean,
            "std": out_std,
            "nodata_meta": out_meta_nodata[i],
            "nodata_used": nodata_value,
        }
        out_settings[current_transform] = current_settings

    return out_array, out_meta, out_settings


def _minmax_scaling_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    new_range: List[Tuple[int | float, int | float]] = [(0, 1)],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
    raster = in_data

    if not bands:
        bands = list(range(1, raster.count + 1))

    expanded_args = utils.expand_args(selection=bands, nodata=nodata, new_range=new_range)
    nodata = expanded_args["nodata"]
    new_range = expanded_args["new_range"]

    out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
    out_settings = {}

    for i, band_idx in enumerate(bands_idx):
        nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]

        out_array[band_idx] = _minmax_scaling_core(
            data_array=out_array[band_idx], new_range=new_range[i], nodata_value=nodata_value
        )

        current_transform = f"band {band_idx + 1}"
        current_settings = {
            "band_origin": bands[i],
            "scaled_min": new_range[i][0],
            "scaled_max": new_range[i][1],
            "nodata_meta": out_meta_nodata[i],
            "nodata_used": nodata_value,
        }
        out_settings[current_transform] = current_settings

    return out_array, out_meta, out_settings


def z_score_norm(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    with_mean: List[bool] = [True],
    with_sd: List[bool] = [True],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
    """Z-score normalization.

    Transforms input data based on mean and standard deviation.

    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.

    If infinity values occur, they will be replaced by NaN.

    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.

    If only one NoData, with_mean or with_sd value is specified, it will be used for all (selected) bands.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).

    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order in the output corresponds to the order of the specified selection.

    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int], optional): Bands or columns to be processed. Defaults to None.
        with_mean (List[bool]): If True, data-based mean will be used, otherwise mean = 0. Defaults to True.
        with_sd (List[bool]): If True, data-based standard deviatioin will be used, otherwise std = 1. Defaults to True.
        nodata (List[int | float], optional): NoData values to be considered. Defaults to None.
        method (Literal["replace", "extract"]): Switch for data output. Defaults to "replace".

    Returns:
        out_array (np.ndarray): The transformed data.
        out_meta (dict): Updated metadata with new band count.
        out_settings (dict): Return of the input settings related to the new ordered output.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """
    valids = parameter.check_selection(in_data, selection)
    valids.append(("With mean length", parameter.check_parameter_length(selection, with_mean, choice=1)))
    valids.append(("With std length", parameter.check_parameter_length(selection, with_sd, choice=1)))
    valids.append(("NoData length", parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))
    valids.append(("With mean data type", all(isinstance(item, bool) for item in with_mean)))
    valids.append(("With std data type", all(isinstance(item, bool) for item in with_sd)))

    if nodata is not None:
        valids.append(("NoData data type", all(isinstance(item, Union[int, float, None]) for item in nodata)))

    if isinstance(in_data, rasterio.DatasetReader):
        valids.append(("Output method", method == "replace" or method == "extract"))

        for item in valids:
            error_msg, validation = item

            if validation == False:
                raise InvalidParameterValueException(error_msg)

        out_array, out_meta, out_settings = _z_score_normalization_raster(
            in_data=in_data, bands=selection, with_mean=with_mean, with_sd=with_sd, nodata=nodata, method=method
        )

        return out_array, out_meta, out_settings


def minmax_scaling(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    new_range: List[Tuple[int | float, int | float]] = [(0, 1)],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
    """Min-max scaling.

    Transforms input data based on specified new min and maximum values.

    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.

    If infinity values occur, they will be replaced by NaN.

    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.

    If only one NoData value or range tuple is specified, it will be used for all (selected) bands.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).

    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order in the output corresponds to the order of the specified selection.

    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int], optional): Bands or columns to be processed. Defaults to None.
        new_range: (List[Tuple[int | float, int | float]]): List containing the range tuple (min, max) for new minimum and maximum. Defaults to (0, 1).
        nodata (List[int | float], optional): NoData values to be considered. Defaults to None.
        method (Literal["replace", "extract"]): Switch for data output. Defaults to "replace".

    Returns:
        out_array (np.ndarray): The transformed data.
        out_meta (dict): Updated metadata with new band count.
        out_settings (dict): Return of the input settings related to the new ordered output.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """
    valids = parameter.check_selection(in_data, selection)
    valids.append(("New range length", parameter.check_parameter_length(selection, new_range, choice=1)))
    valids.append(("NoData length", parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))
    valids.append(
        (
            "New range values data type",
            min([all(isinstance(element, Union[int, float]) for element in item) for item in new_range]),
        )
    )
    valids.append(
        (
            "New range values length",
            all(parameter.check_parameter_length(parameter=item, choice=2) for item in new_range),
        )
    )
    valids.append(("New range values order", all(parameter.check_minmax_position(item) for item in new_range)))

    if nodata is not None:
        valids.append(("NoData data type", all(isinstance(item, Union[int, float, None]) for item in nodata)))

    if isinstance(in_data, rasterio.DatasetReader):
        valids.append(("Output method", method == "replace" or method == "extract"))

        for item in valids:
            error_msg, validation = item

            if validation == False:
                raise InvalidParameterValueException(error_msg)

        out_array, out_meta, out_settings = _minmax_scaling_raster(
            in_data=in_data, bands=selection, new_range=new_range, nodata=nodata, method=method
        )

        return out_array, out_meta, out_settings
