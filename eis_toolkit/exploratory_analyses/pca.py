from numbers import Number
from typing import Literal, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from beartype import beartype
from beartype.typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from eis_toolkit import exceptions

SCALERS = {"standard": StandardScaler, "min_max": MinMaxScaler, "robust": RobustScaler}


@beartype
def _prepare_array_data(
    feature_matrix: np.ndarray, nodata_value: Optional[Number] = None, reshape: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if reshape:
        bands, rows, cols = feature_matrix.shape
        feature_matrix = feature_matrix.transpose(1, 2, 0).reshape(rows * cols, bands)

    if feature_matrix.size == 0:
        raise exceptions.EmptyDataException("Input data is empty.")

    feature_matrix, missing_values_mask = _handle_missing_values(feature_matrix, nodata_value)

    return feature_matrix, missing_values_mask


@beartype
def _handle_missing_values(
    feature_matrix: np.ndarray, nodata_value: Optional[Number] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if nodata_value is not None:
        feature_matrix[feature_matrix == nodata_value] = np.nan
    missing_values_mask = np.isnan(feature_matrix)
    feature_matrix[missing_values_mask] = 0  # Mask nan to 0 for processing

    return feature_matrix, missing_values_mask


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
    data: Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame, rasterio.io.DatasetReader],
    number_of_components: int,
    scaler_type: Literal["standard", "min_max", "robust"] = "standard",
    nodata: Optional[Number] = None,
    color_column_name: Optional[str] = None,
) -> Tuple[
    Union[
        np.ndarray,
        Tuple[pd.DataFrame, sns.PairGrid],
        Tuple[gpd.GeoDataFrame, sns.PairGrid],
        Tuple[np.ndarray, rasterio.profiles.Profile],
    ],
    np.ndarray,
]:
    """
    Compute given number of principal components for numeric input data.

    Various input data formats are accepted and the output format depends on the input format. If
    input is (Geo)DataFrame, a pairplot is produced additionally. A column name used for coloring can
    be specified in this case.

    Args:
        data: Input data for PCA.
        number_of_components: The number of principal components to compute Should be >= 1 and at most
            the number of numeric columns if input is (Geo)DataFrame or number of bands if input is raster.
        scaler_type: Transform data according to a specified Sklearn scaler.
            Options are "standard", "min_max" and "robust". Defaults to "standard".
        nodata: Define nodata value to be masked out. Optional parameter. If None and input is raster, looks
            for nodata value from raster metadata. Defaults to None.
        color_column_name: If input data is a DataFrame or a GeoDataFrame, column name used for
            coloring data points in the produced pairplot can be defined. Defaults to None.

    Returns:
        The computed principal components in corresponding format as the input data (for raster, output is
        Numpy array containing the data and raster profile) and the explained variance ratios for each component.

    Raises:
        EmptyDataException: The input is empty.
        InvalidNumberOfPrincipalComponents: The number of principal components is less than 1 or more than
            number of columns if input was (Geo)DataFrame.
    """
    if scaler_type not in SCALERS:
        raise exceptions.InvalidParameterValueException(f"Invalid scaler. Choose from: {list(SCALERS.keys())}")

    if number_of_components < 1:
        raise exceptions.InvalidParameterValueException("The number of principal components should be >= 1.")

    # Get feature matrix (Numpy array) from various input types
    if isinstance(data, np.ndarray):
        feature_matrix = data
        if feature_matrix.ndim == 2:  # Table-like data (assumme it is a DataFrame transformed to Numpy array)
            feature_matrix, nan_mask = _prepare_array_data(feature_matrix, nodata_value=nodata, reshape=False)
        elif feature_matrix.ndim == 3:  # Assume data represents multiband raster data
            rows, cols = feature_matrix.shape[1], feature_matrix.shape[2]
            feature_matrix, nan_mask = _prepare_array_data(feature_matrix, nodata_value=nodata, reshape=True)
        else:
            raise exceptions.InvalidParameterValueException(
                f"Unsupported input data format. {feature_matrix.ndim} dimensions detected."
            )
        if feature_matrix.size == 0:
            raise exceptions.EmptyDataException("Input array is empty.")

    elif isinstance(data, rasterio.io.DatasetReader):
        feature_matrix = data.read()
        if feature_matrix.ndim < 3:
            raise exceptions.InvalidParameterValueException("Input raster should have multiple bands.")
        rows, cols = feature_matrix.shape[1], feature_matrix.shape[2]
        if nodata is None:
            nodata = data.nodata
        feature_matrix, nan_mask = _prepare_array_data(feature_matrix, nodata_value=nodata, reshape=True)

    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if df.empty:
            raise exceptions.EmptyDataException("Input DataFrame is empty.")
        if number_of_components > len(df.columns):
            raise exceptions.InvalidParameterValueException(
                "The number of principal should be at most the number of numeric columns in the input DataFrame."
            )
        if color_column_name is not None:
            color_column_data = df[color_column_name]

        if isinstance(data, gpd.GeoDataFrame):
            geometries = data.geometry
            crs = data.crs
            df = df.drop(columns=["geometry"])

        df = df.convert_dtypes()
        df = df.apply(pd.to_numeric, errors="ignore")
        df = df.select_dtypes(include=np.number)
        df = df.astype(dtype=np.number)
        feature_matrix = df.to_numpy()
        feature_matrix = feature_matrix.astype(float)
        feature_matrix, nan_mask = _handle_missing_values(feature_matrix, nodata)

    # Core PCA computation
    principal_components, explained_variances = _compute_pca(feature_matrix, number_of_components, scaler_type)

    # Put nodata back in and consider new dimension of data
    if nodata is not None:
        principal_components[nan_mask[:, number_of_components]] = nodata
    else:
        principal_components[nan_mask[:, :number_of_components]] = np.nan

    # Convert PCA output to proper format
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            result_data = principal_components.reshape(rows, cols, -1).transpose(2, 0, 1)
        else:
            result_data = principal_components

    elif isinstance(data, rasterio.io.DatasetReader):
        principal_components = principal_components.reshape(rows, cols, -1).transpose(2, 0, 1)
        out_profile = data.profile.copy()
        out_profile["count"] = number_of_components
        out_profile["dtype"] = "float32"
        result_data = (principal_components, out_profile)

    elif isinstance(data, pd.DataFrame):
        component_names = [f"principal_component_{i+1}" for i in range(number_of_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=component_names)
        if color_column_name is not None:
            pca_df[color_column_name] = color_column_data
        sns_pair_grid = plot_pca(pca_df, explained_variances, color_column_name)
        if isinstance(data, gpd.GeoDataFrame):
            pca_df = gpd.GeoDataFrame(pca_df, geometry=geometries, crs=crs)
        result_data = (pca_df, sns_pair_grid)

    return result_data, explained_variances


@beartype
def plot_pca(
    pca_df: pd.DataFrame,
    explained_variances: Optional[np.ndarray] = None,
    color_column_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> sns.PairGrid:
    """Plot a scatter matrix of different principal component combinations.

    Args:
        pca_df: A DataFrame containing computed principal components.
        explained_variances: The explained variance ratios for each principal component. Used for labeling
            axes in the plot. Optional parameter. Defaults to None.
        color_column: Name of the column that will be used for color-coding data points. Typically a
            categorical variable in the original data. Optional parameter, no colors if not provided.
            Defaults to None.
        save_path: The save path for the plot. Optional parameter, no saving if not provided. Defaults to None.

    Returns:
        A Seaborn pairgrid containing the PCA scatter matrix.

    Raises:
        InvalidColumnException: DataFrame does not contain the given color column.
    """

    if color_column_name and color_column_name not in pca_df.columns:
        raise exceptions.InvalidColumnException("DataFrame does not contain the given color column.")

    pair_grid = sns.pairplot(pca_df, hue=color_column_name)

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
