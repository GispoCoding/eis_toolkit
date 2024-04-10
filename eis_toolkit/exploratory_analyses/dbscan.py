from numbers import Number

import geopandas as gdp
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Sequence
from sklearn.cluster import DBSCAN

from eis_toolkit.exceptions import (
    EmptyDataException,
    EmptyDataFrameException,
    InvalidColumnException,
    InvalidDataShapeException,
    InvalidParameterValueException,
)


def _dbscan(data_matrix: np.ndarray, max_distance: float, min_samples: int) -> DBSCAN:
    dbscan = DBSCAN(eps=max_distance, min_samples=min_samples)
    dbscan.fit(data_matrix)
    return dbscan


@beartype
def dbscan_vector(
    data: gdp.GeoDataFrame,
    include_coordinates: bool = True,
    columns: Optional[Sequence[str]] = None,
    max_distance: Number = 0.5,
    min_samples: int = 5,
) -> gdp.GeoDataFrame:
    """
    Perform DBSCAN clustering on a Geodataframe.

    The attributes to include in clustering can be controlled with `include_coordinates` and
    `columns` parameters. Coordinates will add spatial proximity and columns the selected
    attributes in the cluster creation process. If coordinates are omitted, at least some columns
    need to be included.

    If columns are included and the attributes have different scales and represent different
    phenomena, consider normalizing or standardizing data before running DBSCAN to avoid biased clusters.

    Note that the results depend heavily on the parameter values that might require careful tuning.
    Note also that clustering can be computationally intesive for large datasets, for highly dimensional data
    consider dimensionality reduction techniques such as PCA.

    Args:
        data: GeoDataFrame containing the input data.
        include_coordinates: If feature coordinates (spatial proximity) will be included in
            the clustering process. Defaults to True.
        columns: Columns/attributes in the input Geodataframe to be included in the clustering
            process. Optional parameter, defaults to no columns included (except coordinates).
        max_distance: The maximum distance between two samples for one to be considered as in the neighborhood of
            the other. Defaults to 0.5.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            Defaults to 5.

    Returns:
        GeoDataFrame containing new column for assigned cluster labels.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidColumnException: All specified columns were not found in the input GeoDataFrame.
        InvalidParameterException: The maximum distance between two samples in a neighborhood is not greater
            than zero, the number of samples in a neighborhood is not greater than one or or both coordinates
            and attributes are omitted.
    """
    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if max_distance <= 0:
        raise InvalidParameterValueException(
            "The input value for the maximum distance between two samples in a neighborhood must be greater than zero."
        )

    if min_samples <= 1:
        raise InvalidParameterValueException(
            "The input value for the minimum number of samples in a neighborhood must be greater than one."
        )

    if columns is None or len(columns) == 0:
        if not include_coordinates:
            raise InvalidParameterValueException(
                "No attributes or coordinates included, cannot perform DBSCAN clustering."
            )
    else:
        invalid_columns = [column for column in columns if column not in data.columns]
        if invalid_columns:
            raise InvalidColumnException(f"Invalid columns: {invalid_columns}")

    if include_coordinates:
        coordinates = np.array(list(data.geometry.apply(lambda geom: [geom.x, geom.y])))
    else:
        coordinates = np.empty((len(data), 0))

    if columns:
        attributes = data[list(columns)].to_numpy()
        data_matrix = np.hstack((coordinates, attributes)) if include_coordinates else attributes
    else:
        data_matrix = coordinates

    dbscan = _dbscan(data_matrix, max_distance, min_samples)

    result_gdf = data.copy()
    result_gdf["cluster"] = dbscan.labels_
    return result_gdf


@beartype
def dbscan_array(data: np.ndarray, max_distance: Number = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Perform DBSCAN clustering on Numpy array data.

    If the bands/datasets that form the input 3D Numpy array have different scales and represent different
    phenomena, consider normalizing or standardizing data before running DBSCAN to avoid biased clusters.

    Note that the results depend heavily on the parameter values that might require careful tuning.
    Note also that clustering can be computationally intesive for large datasets, for highly dimensional data
    consider dimensionality reduction techiniques such as PCA.

    Args:
        data: A 3D Numpy array containing the input data. Expects data to be stacked 2D arrays
            with shape (bands, height, width).
        max_distance: The maximum distance between two samples for one to be considered as in the neighborhood of
            the other. Defaults to 0.5.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            Defaults to 5.

    Returns:
        Clustering results as a 2D cluster labels array.

    Raises:
        EmptyDataException: The input Numpy array is empty.
        InvalidDataShapeException: Input data has incorrect number of dimensions (other than 3).
        InvalidParameterException: The maximum distance between two samples in a neighborhood is not greater
            than zero or the number of samples in a neighborhood is not greater than one.
    """
    if data.size == 0:
        raise EmptyDataException("The input array data is empty.")

    if data.ndim != 3:
        raise InvalidDataShapeException(f"This input array is not 3D: {data.ndim}.")

    if max_distance <= 0:
        raise InvalidParameterValueException(
            "The input value for the maximum distance between two samples in a neighborhood must be greater than zero."
        )

    if min_samples <= 1:
        raise InvalidParameterValueException(
            "The input value for the minimum number of samples in a neighborhood must be greater than one."
        )

    n_bands, height, width = data.shape
    reshaped_data = data.reshape(n_bands, height * width).T  # Transpose to get correct shape
    mask = np.any(np.isnan(reshaped_data), axis=1)  # Filter NaN if present
    data_matrix = reshaped_data[~mask]

    dbscan = _dbscan(data_matrix, max_distance, min_samples)

    labels = np.full((height * width), -9999, dtype=int)
    labels[~mask] = dbscan.labels_
    labels_image = labels.reshape((height, width))
    return labels_image
