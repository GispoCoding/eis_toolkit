import rasterio
import pandas as pd
import geopandas as gpd
from typing import Optional, Tuple, Union, Sequence, List
from numbers import Number


def check_parameter_value(parameter_value: Union[Number, str], allowed_values: Union[list, tuple]) -> bool:
    """
    Check if used parameter value is valid.
    Args:
        parameter_value: value given to a function.
        allowed_values: sequence of values containing allowed parameter values.
    Returns:
        Bool: True if parameter value is allowed, False if not.
    """
    return parameter_value in allowed_values


def check_numeric_value_sign(parameter_value) -> bool:  # type: ignore[no-untyped-def]
    """
    Check if input numeric value is positive.
    Args:
        parameter_value: numeric input parameter.
    Returns:
        Bool: True if parameter value is positive, False if not.
    """
    return True if parameter_value > 0 else False


def check_dtype_for_int(scalar: Number) -> bool:
    """
    Determines whether a floating scalar can be converted to integer type.
    Args:
        scalar: Input scalar value.
    Returns:
        True if conversion can be done, False if not.
    """
    return True if isinstance(scalar, int) else scalar.is_integer()


def check_parameter_length(  # type: ignore[no-untyped-def]
    selection: Optional[Sequence] = None, parameter: Optional[Sequence] = None
) -> bool:
    """
    Check the length of a parameter against the length of selected bands.
    Args:
        selection: Selected bands.
        parameter: List containing parameter values.
    Returns:
        Bool: True if conditions are valid, else False.
    """
    return len(parameter) == 1 or len(parameter) == len(selection)


# delete, or at least simplify
def check_none_positions(parameter: Sequence) -> List:  # type: ignore[no-untyped-def]
    """
    Check the position of NoneType values.
    Args:
        parameter_value: list or tuple containing parameter values.
    Returns:
        List: List of position for NoneType values.
    """
    out_positions = []
    for i, inner_value in enumerate(parameter):
        if inner_value is None:
            out_positions.append(i)
        elif isinstance(inner_value, tuple):
            for j, value in enumerate(inner_value):
                if value is None:
                    out_positions.append((i, j))

    return out_positions


def check_numeric_minmax_location(parameter: tuple) -> bool:  # type: ignore[no-untyped-def]
    """Check if parameter maximum value > parameter minimum value.
    Args:
        parameter: Tuple containing parameter values for min and max.
    Returns:
        Bool: True if minimum value < maxiumum value, else False.
    """
    return parameter[0] < parameter[1]


# delete before PR
def check_selection(  # type: ignore[no-untyped-def]
    in_data: rasterio.DatasetReader,
    selection: Optional[Sequence[int]] = None,
    validation_results: Optional[Sequence[tuple]] = None,
) -> list[tuple]:
    """Checks whether the selection of raster bands or is valid or not.

    Args:
        in_data: raster object or pandas dataframe/geopandas geodataframe.
        selection: selected bands/columns.
        validation_results: list containing a tuple for existing case and validation results.

    Returns:
        List: list containing a tuple for case and validation results.
    """
    if validation_results is None:
        validation_results = []

    if isinstance(in_data, rasterio.DatasetReader):
        if selection is not None:
            validation = all([isinstance(item, int) for item in selection])
            validation_results.append(("Band selection value data type", validation))  # beartype
            validation_results.append(
                ("Band selection value zero", 0 not in selection)
            )  # accomplished in check_raster_bands
            validation_results.append(
                ("Band selection values not unique", len(set(selection)) == len(selection))
            )  # check discarded. if a band is selected multiple times, it can be thresholded with different values
            validation_results.append(("Band selection length", len(selection) <= in_data.count))

            if None not in selection and validation == True:
                validation_results.append(("Band selection maximum value", max(selection) <= in_data.count))

    return validation_results
