import os
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio


def _read_label_raster(file):
    """Read a a raster file with label data."""
    with rasterio.open(file) as src:
        label_raster = src.read(1)  # Assuming labels are in the first band
        nodata_value = src.nodata
    return label_raster, nodata_value


def _read_and_stack_training_raster(file):
    """Read all bands of raster file with training data in a stack."""
    with rasterio.open(file) as src:
        raster_data = np.stack([src.read(i) for i in range(1, src.count + 1)])
        profile = src.profile
    return raster_data, profile


def _reshape_training_data_for_ml_and_mask(rasters, nodata_values):
    """Reshape training data and mask nodata cells outs."""
    reshaped_data = []
    mask = None

    for raster, nodata in zip(rasters, nodata_values):
        raster_reshaped = raster.reshape(raster.shape[0], -1).T
        reshaped_data.append(raster_reshaped)

        if nodata is not None:
            raster_mask = (raster_reshaped == nodata).any(axis=1)
            mask = raster_mask if mask is None else mask | raster_mask

    X = np.concatenate(reshaped_data, axis=1)

    return X, mask


def prepare_data_for_ml(
    training_raster_files: Sequence[Union[str, os.PathLike]],
    label_file: Optional[Union[str, os.PathLike]] = None,
    labels_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, Any]:
    """Prepare data ready for machine learning model training.

    Performs the following steps:
    - Read all bands of training rasters into a stacked array
    - Reshape training data
    - Read label data (rasterize if a vector file is given)
    - Create a nodata mask using all training rasters and labels, and mask nodata cells out
    - Return the read, reshaped and masked data as (X, y) and reference raster profile (first training raster)
    """
    # Read and stack training rasters
    training_data, profiles = zip(*[_read_and_stack_training_raster(file) for file in training_raster_files])
    nodata_values = [profile["nodata"] for profile in profiles]
    # original_shape = training_data[0].shape[1:]

    # Reshape training data for ML and create mask
    X, feature_mask = _reshape_training_data_for_ml_and_mask(training_data, nodata_values)

    if label_file is not None:
        # Check label file type and process accordingly
        file_extension = os.path.splitext(label_file)[1].lower()
        if file_extension in [".shp", ".geojson", ".json"]:  # Vector data
            pass
            # TODO: Use rasterize from EIS toolkit
            # y, label_mask = rasterize_vector_data(label_file, labels_column, training_data[0])
        else:  # Raster data
            y, label_nodata = _read_label_raster(label_file)
            label_mask = y == label_nodata

        # Combine masks and apply to feature and label data
        combined_mask = feature_mask | label_mask.ravel()

        y = y.ravel()[~combined_mask]

    else:
        combined_mask = feature_mask
        y = None

    X = X[~combined_mask]

    return X, y, profiles[0], combined_mask
