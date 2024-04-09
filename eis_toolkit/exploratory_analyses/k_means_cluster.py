import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Sequence
from sklearn.cluster import KMeans

from eis_toolkit.exceptions import (
    EmptyDataException,
    EmptyDataFrameException,
    InvalidColumnException,
    InvalidDataShapeException,
    InvalidParameterValueException,
)


def _k_means_clustering(
    data_matrix: np.ndarray,
    number_of_clusters: Optional[int],
    random_state: Optional[int],
) -> KMeans:
    if number_of_clusters is None:
        # The elbow method
        k_max = 10
        inertia = np.array(
            [KMeans(n_clusters=k, random_state=0, n_init=10).fit(data_matrix).inertia_ for k in range(1, k_max + 1)]
        )
        inertia = np.diff(inertia, 2)
        scaled_derivatives = [i * 100 for i in inertia]
        number_of_clusters = scaled_derivatives.index(min(scaled_derivatives))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state, n_init=10)
    kmeans.fit(data_matrix)

    return kmeans


def k_means_clustering_array(
    data: np.ndarray,
    number_of_clusters: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Perform k-means clustering on Numpy array data.

    If the bands/datasets that form the input 3D Numpy array have different scales and represent different
    phenomena, consider normalizing or standardizing data before running k-means to avoid biased clusters.

    Args:
        data: A 3D Numpy array containing the input data. Expects data to be stacked 2D arrays
            with shape (bands, height, width).
        number_of_clusters: The number of clusters (>= 1) to form. Optional parameter. If not provided,
            optimal number of clusters is computed using the elbow method.
        random_state: A random number generation for centroid initialization to make
            the randomness deterministic. Optional parameter.

    Returns:
        Clustering results as a 2D cluster labels array.

    Raises:
        EmptyDataException: The input Numpy array is empty.
        InvalidDataShapeException: Input data has incorrect number of dimensions (other than 3).
        InvalidParameterException: The number of clusters is less than one.
    """
    if data.size == 0:
        raise EmptyDataException("The input raster data is empty.")

    if data.ndim != 3:
        raise InvalidDataShapeException(f"The input array is not 3D: {data.ndim}.")

    if number_of_clusters is not None and number_of_clusters < 1:
        raise InvalidParameterValueException("The input value for number of clusters must be at least one.")

    n_bands, height, width = data.shape
    reshaped_data = data.reshape(n_bands, height * width).T  # Transpose to get correct shape
    mask = np.any(np.isnan(reshaped_data), axis=1)  # Filter NaN if present
    data_matrix = reshaped_data[~mask]

    kmeans = _k_means_clustering(data_matrix, number_of_clusters, random_state)

    labels = np.full((height * width), -9999, dtype=int)
    labels[~mask] = kmeans.labels_
    labels_image = labels.reshape((height, width))
    return labels_image


@beartype
def k_means_clustering_vector(
    data: gpd.GeoDataFrame,
    include_coordinates: bool = True,
    columns: Optional[Sequence[str]] = None,
    number_of_clusters: Optional[int] = None,
    random_state: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Perform k-means clustering on a Geodataframe.

    The attributes to include in clustering can be controlled with `include_coordinates` and
    `columns` parameters. Coordinates will add spatial proximity and columns the selected
    attributes in the cluster creation process. If coordinates are omitted, at least some columns
    need to be included.

    If columns are included and the attributes have different scales and represent different
    phenomena, consider normalizing or standardizing data before running k-means to avoid biased clusters.

    Args:
        data: A GeoDataFrame containing the input data.
        include_coordinates: If feature coordinates (spatial proximity) will be included in
            the clustering process. Defaults to True.
        columns: Columns/attributes in the input Geodataframe to be included in the clustering
            process. Optional parameter, defaults to no columns included (except coordinates).
        number_of_clusters: The number of clusters (>= 1) to form. Optional parameter. If not provided,
            optimal number of clusters is computed using the elbow method.
        random_state: A random number generation for centroid initialization to make
            the randomness deterministic. Optional parameter.

    Returns:
        GeoDataFrame containing assigned cluster labels.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterException: The number of clusters is less than one or both coordinates and attributes
            are omitted.
        InvalidColumnException: All specified columns were not found in the input GeoDataFrame.
    """
    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if number_of_clusters is not None and number_of_clusters < 1:
        raise InvalidParameterValueException("The input value for number of clusters must be at least one.")

    if columns is None or len(columns) == 0:
        if not include_coordinates:
            raise InvalidParameterValueException(
                "No attributes or coordinates included, cannot perform k-means clustering."
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

    kmeans = _k_means_clustering(data_matrix, number_of_clusters, random_state)

    result_gdf = data.copy()
    result_gdf["cluster"] = kmeans.labels_
    return result_gdf
