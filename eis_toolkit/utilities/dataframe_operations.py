from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from eis_toolkit.checks.argument_types import check_argument_types


def get_columns_by_type(fields: dict, column_types: Tuple[str, ...]) -> set:
    return {column for column, col_type in fields.items() if col_type in column_types}


@check_argument_types
def separation(df: pd.DataFrame, fields: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separates the target column (id exists) to a separate dataframe ydf
    All categorical columns (fields) will be separated from all other features (columns) in a separate dataframe Xcdf.
    Separates the id and geometry column to a separate dataframe igdf

    Args:
        df (pandas DataFrame): Including target column ('t').
        fields (dictionary): Column type for each column

    Returns:
        pandas DataFrame: value-sample  (Xvdf)
        pandas DataFrame: categorical columns (Xcdf)
        pandas DataFrame: target (ydf)
        pandas DataFrame: target (igdf)
    """

    column_names = df.columns
    ydf_columns = get_columns_by_type(fields, ("t",))
    Xvdf_columns = get_columns_by_type(fields, ("v", "b"))
    Xcdf_columns = get_columns_by_type(fields, ("c",))
    igdf_columns = get_columns_by_type(fields, ("i", "g"))

    for col_set in (ydf_columns, Xvdf_columns, Xcdf_columns, igdf_columns):
        if not col_set.issubset(column_names):
            raise Exception("fields and column names of DataFrame df do not match")

    ydf = df[list(ydf_columns)]
    Xvdf = df[list(Xvdf_columns)]
    Xcdf = df[list(Xcdf_columns)]
    igdf = df[list(igdf_columns)]

    return Xvdf, Xcdf, ydf, igdf


def extract_target_from_features(features_df: pd.DataFrame, fields: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_field = {i for i in fields if fields[i] == "t"}
    if not set(target_field).issubset(set(features_df.columns)):
        raise Exception("fields and column names of DataFrame features_df do not match")
    target_df = features_df[target_field]
    features_df.drop(target_field, axis=1, inplace=True)
    return features_df, target_df


@check_argument_types
def split(
    features_df: pd.DataFrame,
    target_df: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
    test_size: Optional[int | float] = None,
    train_size: Optional[int | float] = None,
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if target_df is None and fields is not None:
        features_df, target_df = extract_target_from_features(features_df, fields)

    if test_size is None and train_size is None:
        return features_df, pd.DataFrame(), target_df, pd.DataFrame()

    if target_df is not None:
        train_features, test_features, train_target, test_target = train_test_split(
            features_df,
            target_df,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    else:
        train_features, test_features = train_test_split(
            features_df, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle
        )
        train_target = None
        test_target = None

    return train_features, test_features, train_target, test_target


@check_argument_types
def one_hot_encoder(
    df: pd.DataFrame, ohe: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder]]:
    """
    Encodes all categorical columns in a pandas dataframe.
    In case of model training: onehotencoder object is one of the outputs.
    In case of prediction: onehotencoder object created in traing is needed (input Argument).

    Args:
        - df (DataFrame): contains all c-typed columns which should not be float
        - ohe: in case of prediction mandatory
               in case of training = None

    Returns:
        pandas DataFrame: binarized
        ohe - Object (OneHotEncoding): in case of training
                                       in case of prediction: None
    """
    if ohe is None:
        ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False, dtype=int)
        ohe.fit(df)

    transformed = ohe.transform(df)
    transformed_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out())

    return transformed_df, ohe


def transform_fit_dataframe(df):
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False, dtype=int)
    ohe.fit(df)
    return transform_dataframe(df, ohe)


def transform_dataframe(df: pd.DataFrame, ohe: OneHotEncoder):
    transformed = ohe.transform(df)
    return pd.DataFrame(transformed, columns=ohe.get_feature_names_out())


@check_argument_types
def one_hot_encoder_pandas(input_df: pd.DataFrame, encoded_columns: List[str]) -> pd.DataFrame:

    encoded_df = pd.get_dummies(data=input_df, columns=encoded_columns, dummy_na=True)

    return encoded_df
