import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Union

from eis_toolkit.exceptions import InvalidDataShapeException, InvalidParameterValueException


@beartype
def standardize(
    data: Union[np.ndarray, pd.DataFrame],
    array_type: Literal["tabular", "raster"] = "tabular",
    columns: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Standardize input data.

    Standardization is applied to each variable independently. For DataFrames, each column is
    treated as a variable and for Numpy arrays, either each column or each 2D array is treated
    as a variable based on `array_type` value.

    Args:
        data: Input data to be standardized, either a numpy array or a pandas DataFrame.
        array_type: Specifies how the data is interpreted if input is numpy array.
            `tabular` is used for 2D data where each column is a variable (data preparation for ML modeling),
            and `raster` for 2D/3D data where 2D array is a variable.
        columns: Column selection if input is a DataFrame, ignored if input is numpy array.

    Returns:
        Standardized data in the input format.

    Raises:
        InvalidParameterValueException: If array type selection is invalid.
        InvalidDataShapeException: If shape of Numpy array is invalid for selected array type.
    """
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        for col in columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    else:
        if array_type == "tabular":
            if data.ndim != 2:
                raise InvalidDataShapeException("Tabular data must be a 2D numpy array.")
            data = (data - data.mean(axis=0)) / data.std(axis=0)
        elif array_type == "raster":
            if data.ndim == 2:  # Treat like a single-band raster
                data = (data - data.mean()) / data.std()
            elif data.ndim == 3:
                for i in range(data.shape[0]):
                    data[i, :, :] = (data[i, :, :] - data[i, :, :].mean()) / data[i, :, :].std()
            else:
                raise InvalidDataShapeException("Raster data must be a 2D or 3D numpy array.")
        else:
            raise InvalidParameterValueException("array_type must be either 'tabular' or 'raster'.")
    return data
