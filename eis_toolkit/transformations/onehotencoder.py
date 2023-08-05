import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from sklearn.preprocessing import OneHotEncoder

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _onehotencoder(
    df: pd.DataFrame,
    ohe: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Union[OneHotEncoder, None]]:

    encnew = None
    if len(df.columns) == 0 and len(df.index) == 0:
        tmpb = None
    else:
        if ohe is not None:
            tmpb = ohe.transform(df)
            tmpb = pd.DataFrame(tmpb, columns=ohe.get_feature_names_out())
        else:
            if len((df.select_dtypes(include=[float])).columns) > 0:
                raise InvalidParameterValueException("One of the c-type fields is float")
            encnew = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False, dtype=int)
            encnew.fit(df)
            tmpb = encnew.transform(df)
            tmpb = pd.DataFrame(tmpb, columns=encnew.get_feature_names_out())

    return tmpb, encnew


# *******************************
@beartype
def onehotencoder(
    df: pd.DataFrame, ohe: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Union[OneHotEncoder, None]]:
    """
        Encode all categorical columns in a pandas dataframe to binary columns (0/1).

        In case of model training: onehotencoder object is one of the outputs.
        In case of prediction: onehotencoder object created in traing is needed (input Parameter).
           On this way the same binary columns as in training process will be created.
    Args:
        - df: cantains all c-typed columns witch should not be float
        - ohe:  in case of predition mandantory
                in case of training = None
    Returns:
        binarized dataframe
        OnHotEncoing-Object:    in case of training
                                in case of prediction: None
    """

    return _onehotencoder(df=df, ohe=ohe)
