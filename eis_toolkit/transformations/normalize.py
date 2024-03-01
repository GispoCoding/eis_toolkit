import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Union

from eis_toolkit.exceptions import InvalidDataShapeException, InvalidParameterValueException


@beartype
def normalize(
    data: Union[np.ndarray, pd.DataFrame],
    array_type: Literal["tabular", "raster"] = "tabular",
    columns: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Normalize input data.

    Normalization is applied to each variable independently. For DataFrames, each column is
    treated as a variable and for Numpy arrays, either each column or each 2D array is treated
    as a variable based on `array_type` value.

    Args:
        data: Input data to be normalized, either a numpy array or a pandas DataFrame.
        array_type: Specifies how the data is interpreted if input is numpy array.
            `tabular` is used for 2D data where each column is a variable (data preparation for ML modeling),
            and `raster` for 2D/3D data where 2D array is a variable.
        columns: Column selection if input is a DataFrame, ignored if input is numpy array.

    Returns:
        Normalized data in the input format.

    Raises:
        InvalidParameterValueException: If array type selection is invalid.
        InvalidDataShapeException: If shape of Numpy array is invalid for selected array type.
    """
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        for col in columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    else:
        if array_type == "tabular":
            if data.ndim != 2:
                raise InvalidDataShapeException("Tabular data must be a 2D numpy array.")
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        elif array_type == "raster":
            if data.ndim == 2:  # Treat like a single-band raster
                data = (data - data.min()) / (data.max() - data.min())
            elif data.ndim == 3:
                for i in range(data.shape[0]):
                    data[i, :, :] = (data[i, :, :] - data[i, :, :].min()) / (data[i, :, :].max() - data[i, :, :].min())
            else:
                raise InvalidDataShapeException("Raster data must be a 2D or 3D numpy array.")
        else:
            raise InvalidParameterValueException("data_type must be either 'tabular' or 'raster'.")
    return data
