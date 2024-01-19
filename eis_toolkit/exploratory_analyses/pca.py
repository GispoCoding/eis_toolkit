from numbers import Number

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from eis_toolkit.exceptions import EmptyDataException, InvalidColumnException, InvalidParameterValueException
from eis_toolkit.utilities.checks.dataframe import check_columns_valid

SCALERS = {"standard": StandardScaler, "min_max": MinMaxScaler, "robust": RobustScaler}


@beartype
def _prepare_array_data(
    feature_matrix: np.ndarray, nodata_handling: str, nodata_value: Optional[Number] = None, reshape: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if reshape:
        bands, rows, cols = feature_matrix.shape
        feature_matrix = feature_matrix.transpose(1, 2, 0).reshape(rows * cols, bands)

    if feature_matrix.size == 0:
        raise EmptyDataException("Input data is empty.")

    return _handle_missing_values(feature_matrix, nodata_handling, nodata_value)


@beartype
def _handle_missing_values(
    feature_matrix: np.ndarray, nodata_handling: str, nodata_value: Optional[Number] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    nodata_mask = None

    if nodata_value is not None:
        nodata_mask = feature_matrix == nodata_value
        feature_matrix[nodata_mask] = np.nan

    if nodata_handling == "remove":
        nan_rows_mask = np.isnan(feature_matrix).any(axis=1)
        feature_matrix = feature_matrix[~nan_rows_mask]
        return feature_matrix, nan_rows_mask

    elif nodata_handling == "replace":
        for i in range(feature_matrix.shape[1]):
            column_mask = np.isnan(feature_matrix[:, i])
            column_mean = np.nanmean(feature_matrix[:, i])
            feature_matrix[column_mask, i] = column_mean
        return feature_matrix, None

    else:
        raise InvalidParameterValueException("Invalid nodata_handling value. Choose 'remove' or 'replace'.")


@beartype
def _compute_pca(
    feature_matrix: np.ndarray, number_of_components: int, scaler_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = SCALERS[scaler_type]()
    scaled_data = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=number_of_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variances = pca.explained_variance_ratio_

    return principal_components, explained_variances


@beartype
def compute_pca(
    data: Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame],
    number_of_components: int,
    columns: Optional[Sequence[str]] = None,
    scaler_type: Literal["standard", "min_max", "robust"] = "standard",
    nodata_handling: Literal["remove", "replace"] = "remove",
    nodata: Optional[Number] = None,
) -> Tuple[Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame], np.ndarray]:
    """
    Compute defined number of principal components for numeric input data.

    Before computation, data is scaled according to specified scaler and NaN values removed or replaced.
    Optionally, a nodata value can be given to handle similarly as NaN values.

    If input data is a Numpy array, interpretation of the data depends on its dimensions.
    If array is 3D, it is interpreted as a multiband raster/stacked rasters format (bands, rows, columns).
    If array is 2D, it is interpreted as table-like data, where each column represents a variable/raster band
    and each row a data point (similar to a Dataframe).

    Args:
        data: Input data for PCA.
        number_of_components: The number of principal components to compute. Should be >= 1 and at most
            the number of numeric columns if input is (Geo)Dataframe.
        columns: Select columns used for the PCA. Other columns are excluded from PCA, but added back
            to the result Dataframe intact. Only relevant if input is (Geo)Dataframe. Defaults to None.
        scaler_type: Transform data according to a specified Sklearn scaler.
            Options are "standard", "min_max" and "robust". Defaults to "standard".
        nodata_handling: If observations with nodata (NaN and given `nodata`) should be removed for the time
            of PCA computation or replaced with column/band mean. Defaults to "remove".
        nodata: Define a nodata value to remove. Defaults to None.

    Returns:
        The computed principal components in corresponding format as the input data and the
        explained variance ratios for each component.

    Raises:
        EmptyDataException: The input is empty.
        InvalidColumnException: Selected columns are not found in the input Dataframe.
        InvalidNumberOfPrincipalComponents: The number of principal components is less than 1 or more than
            number of columns if input was (Geo)DataFrame.
        InvalidParameterValueException: If value for `number_of_components` is invalid.
    """
    if scaler_type not in SCALERS:
        raise InvalidParameterValueException(f"Invalid scaler. Choose from: {list(SCALERS.keys())}")

    if number_of_components < 1:
        raise InvalidParameterValueException("The number of principal components should be >= 1.")

    # Get feature matrix (Numpy array) from various input types
    if isinstance(data, np.ndarray):
        feature_matrix = data
        feature_matrix = feature_matrix.astype(float)
        if feature_matrix.ndim == 2:  # Table-like data (assumme it is a DataFrame transformed to Numpy array)
            feature_matrix, nan_mask = _prepare_array_data(
                feature_matrix, nodata_handling=nodata_handling, nodata_value=nodata, reshape=False
            )
        elif feature_matrix.ndim == 3:  # Assume data represents multiband raster data
            rows, cols = feature_matrix.shape[1], feature_matrix.shape[2]
            feature_matrix, nan_mask = _prepare_array_data(
                feature_matrix, nodata_handling=nodata_handling, nodata_value=nodata, reshape=True
            )
        else:
            raise InvalidParameterValueException(
                f"Unsupported input data format. {feature_matrix.ndim} dimensions detected for given array."
            )

    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if df.empty:
            raise EmptyDataException("Input DataFrame is empty.")
        if isinstance(data, gpd.GeoDataFrame):
            geometries = data.geometry
            crs = data.crs
            df = df.drop(columns=["geometry"])
        if columns is not None and columns != []:
            if not check_columns_valid(df, columns):
                raise InvalidColumnException("All selected columns were not found in the input DataFrame.")
            df = df[columns]

        df = df.convert_dtypes()
        df = df.apply(pd.to_numeric, errors="ignore")
        df = df.select_dtypes(include=np.number)
        df = df.astype(dtype=np.number)
        feature_matrix = df.to_numpy()
        feature_matrix = feature_matrix.astype(float)
        feature_matrix, nan_mask = _handle_missing_values(feature_matrix, nodata_handling, nodata)

    if number_of_components > feature_matrix.shape[1]:
        raise InvalidParameterValueException("The number of principal components is too high for the given input data.")
    # Core PCA computation
    principal_components, explained_variances = _compute_pca(feature_matrix, number_of_components, scaler_type)

    if nodata_handling == "remove" and nan_mask is not None:
        principal_components_with_nans = np.full((nan_mask.size, principal_components.shape[1]), np.nan)
        principal_components_with_nans[~nan_mask, :] = principal_components
        principal_components = principal_components_with_nans

    # Convert PCA output to proper format
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            result_data = principal_components.reshape(rows, cols, -1).transpose(2, 0, 1)
        else:
            result_data = principal_components

    elif isinstance(data, pd.DataFrame):
        component_names = [f"principal_component_{i+1}" for i in range(number_of_components)]
        result_data = pd.DataFrame(data=principal_components, columns=component_names)
        if columns is not None:
            old_columns = [column for column in data.columns if column not in columns]
            for column in old_columns:
                result_data[column] = data[column]
        if isinstance(data, gpd.GeoDataFrame):
            result_data = gpd.GeoDataFrame(result_data, geometry=geometries, crs=crs)

    return result_data, explained_variances


@beartype
def plot_pca(
    pca_df: pd.DataFrame,
    explained_variances: Optional[np.ndarray] = None,
    color_column_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> sns.PairGrid:
    """
    Plot a scatter matrix of different principal component combinations.

    Automatically filters columns that do not start with "principal_component" for plotting.
    This tool is designed to work smoothly on `compute_pca` outputs.

    Args:
        pca_df: A DataFrame containing computed principal components.
        explained_variances: The explained variance ratios for each principal component. Used for labeling
            axes in the plot. Optional parameter. Defaults to None.
        color_column_name: Name of the column that will be used for color-coding data points. Typically a
            categorical variable in the original data. Optional parameter, no colors if not provided.
            Defaults to None.
        save_path: The save path for the plot. Optional parameter, no saving if not provided. Defaults to None.

    Returns:
        A Seaborn pairgrid containing the PCA scatter matrix.

    Raises:
        InvalidColumnException: DataFrame does not contain the given color column.
    """

    if color_column_name and color_column_name not in pca_df.columns:
        raise InvalidColumnException("DataFrame does not contain the given color column.")

    filtered_df = pca_df.filter(regex="^principal_component")
    filtered_df = pd.concat([filtered_df, pca_df[[color_column_name]]], axis=1)

    pair_grid = sns.pairplot(filtered_df, hue=color_column_name)

    # Add explained variances to axis labels if provided
    if explained_variances is not None:
        labels = [f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(explained_variances * 100)]
    else:
        labels = [f"PC {i+1}" for i in range(len(pair_grid.axes))]

    # Iterate over axes objects and set the labels
    for i, ax_row in enumerate(pair_grid.axes):
        for j, ax in enumerate(ax_row):
            if j == 0:  # Only the first column
                ax.set_ylabel(labels[i], fontsize="large")
            if i == len(ax_row) - 1:  # Only the last row
                ax.set_xlabel(labels[j], fontsize="large")

    if save_path is not None:
        plt.savefig(save_path)

    return pair_grid
