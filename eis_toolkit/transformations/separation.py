import pandas as pd
from beartype import beartype
from beartype.typing import Tuple

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _separation(
    df: pd.DataFrame, fields: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    cn = df.columns
    # Target dataframe
    name = {i for i in fields if fields[i] == "t"}
    if not set(list(name)).issubset(set(cn)):
        raise InvalidParameterValueException("fields and column names of DataFrame df does not match")
    ydf = df[list(name)]

    # Values dataframe
    name = {i for i in fields if fields[i] in ("v", "b")}
    if not set(list(name)).issubset(set(cn)):
        raise InvalidParameterValueException("fields and column names of DataFrame df does not match")
    Xvdf = df[list(name)]
    # classes dataframe
    name = {i for i in fields if fields[i] == "c"}
    if not set(list(name)).issubset(set(cn)):
        raise InvalidParameterValueException("fields and column names of DataFrame df does not match")
    Xcdf = df[list(name)]

    # identity-geometry dataframe
    name = {i for i in fields if fields[i] in ("i", "g")}
    if not set(list(name)).issubset(set(cn)):
        raise InvalidParameterValueException("fields and column names of DataFrame df does not match")
    igdf = df[list(name)]

    return Xvdf, Xcdf, ydf, igdf


# *******************************
@beartype
def separation(
    df: pd.DataFrame, fields: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

        Separates the target column (id exists) to a separate dataframe ydf.

        All categorical columns (fields) will be separated from all other features (columns)
        in a separate dataframe Xcdf.
        Separates the id and geometry column to a separate dataframe igdf.

    Args:
        - df: Including target column ('t').
        - fields: Column type for each column
            field-types:
            v - values (float or int)
            c - category (int or str)
            t - target (float, int or str)
            b - binery (0 or 1)
            g - geometry
            n - not to use
            i - identifier

    Returns:
        pandas DataFrame with all value and binary columns (not categorical, identification and geometry, 'v' and 'b')
        pandas DataFrame: categorical columns (classes, 'c')
        pandas DataFrame: target column (separated for training process, 't')
        pandas DataFrame: identification and geometry columns ('i' und 'g')
        If the type of columns does not exist the dataframe is empty (no columns)
    """

    # Argument evaluation
    if len(df.columns) == 0:
        raise InvalidParameterValueException("DataFrame has no column")
    if len(df.index) == 0:
        raise InvalidParameterValueException("DataFrame has no rows")
    if len(fields) == 0:
        raise InvalidParameterValueException("Fields is empty")

    # call
    return _separation(df, fields)
