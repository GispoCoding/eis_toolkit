import rasterio
import pandas as pd
import geopandas as gpd
from typing import Optional, Tuple, Union, List, Literal

def check_parameter_value(parameter_value: int, allowed_values: list) -> bool:
    """Check if used parameter value is valid.

    Args:
        parameter_value (int): value given to a function.
        allowed_values (list): a list of allowed parameter values.

    Returns:
        Bool: True if parameter value is allowed, False if not
    """
    if parameter_value in allowed_values:
        return True
    else:
        return False


def check_numeric_value_sign(parameter_value) -> bool:  # type: ignore[no-untyped-def]
    """Check if input numeric value is positive.

    Args:
        parameter_value: numeric input parameter

    Returns:
        Bool: True if parameter value is positive, False if not
    """
    if parameter_value > 0:
        return True
    else:
        return False


def check_none_positions(parameter: Union[List, Tuple]) -> List:   # type: ignore[no-untyped-def]
    """Check the position of NoneType values.

    Args:
        parameter_value: list or tuple containing parameter values

    Returns:
        List: List of position for NoneType values
    """
    out_positions = []
    for i, inner_value in enumerate(parameter):
        if inner_value is None:
            out_positions.append(i)
        elif isinstance(inner_value, Tuple):
            for j, value in enumerate(inner_value):
                if value is None: 
                    out_positions.append((i, j))
        
    return out_positions


def check_parameter_length(  # type: ignore[no-untyped-def]
    selection: Optional[List] = None,
    parameter: Optional[List] = None,
    choice: Optional[int] = None,
    nodata: bool = False,
) -> bool:
    """Check the length of a list against the length of selected bands.

    Args:
        selection: selected bands
        parameter: list containing parameter values
        choice: choice which use case should be validated
        nodata: whether to check parameter or nodata input arguments 

    Returns:
        Bool: True if conditions are valid, else False
    """
    match choice:
        case 1:
            if selection is None or len(selection) == 1:
                if nodata == False:
                    return len(parameter) == 1
                elif nodata == True:
                    return parameter is None or parameter is not None and len(parameter) == 1
            elif selection is not None and len(selection) > 1:
                if nodata == False:
                    return len(parameter) == 1 or len(parameter) == len(selection)
                elif nodata == True:
                    return parameter is None or parameter is not None and len(parameter) == 1 or len(parameter) == len(selection)
        case 2:
            return len(parameter) == 2
        case _:
            return False

        
def check_numeric_minmax_location(parameter: Tuple) -> bool:    # type: ignore[no-untyped-def]
    """Check if parameter maximum value > parameter minimum value.

    Args:
        parameter: tuple containing parameter values for min and max

    Returns:
        Bool: True if minimum value < maxiumum value, else False
    """
    
    if all(isinstance(item, Union[int, float]) for item in parameter):
        return parameter[0] < parameter[1]
    else:
        return False


def check_selection(   # type: ignore[no-untyped-def]
    in_data: rasterio.DatasetReader,
    selection: Optional[List[int | str]] = None,
    validation_results: Optional[List[Tuple]] = None,
) -> List[Tuple]:   
    """Checks whether the selection of raster bands or dataframe columns is valid or not.

    Args:
        in_data: raster object or pandas dataframe/geopandas geodataframe
        selection: selected bands/columns
        validation_results: list containing a tuple for existing case and validation results

    Returns:
        List: list containing a tuple for case and validation results 
    """
    if validation_results is None:
        validation_results = []

    if isinstance(in_data, rasterio.DatasetReader):
        if selection is not None: 
            validation = all([isinstance(item, int) for item in selection])
            validation_results.append(('Band selection value data type', validation))
            validation_results.append(('Band selection value zero', 0 not in selection))
            validation_results.append(('Band selection values not unique', len(set(selection)) == len(selection)))
            validation_results.append(('Band selection length', len(selection) <= in_data.count))
            
            if None not in selection and validation == True: validation_results.append(('Band selection maximum value', max(selection) <= in_data.count))
            
    if isinstance(in_data, Union[pd.DataFrame, gpd.GeoDataFrame]):
        columns_numeric = in_data.select_dtypes(include='number').columns.to_list()

        if selection is None:
            validation_results.append(('No numeric columns', len(columns_numeric) > 0))
        else:
            validation = all([isinstance(item, str) for item in selection])
            validation_results.append(('Selection input data type', validation))
            validation_results.append(('Column selection not unique', len(set(selection)) == len(selection)))
            validation_results.append(('Column selection length', len(selection) <= len(in_data.columns.to_list())))
            validation_results.append(('Column not found', all([column in in_data.columns for column in selection])))

            validation = [column for column in columns_numeric if column in selection]
            validation_results.append(('Non-numeric columns in selection', len(validation) == len(selection)))
            
    return validation_results
