
from typing import Optional, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _onehotencoder(
    Xdf: pd.DataFrame,
    ohe: Optional[Any] = None,
    fields: Optional[Any] = None
) -> pd.DataFrame:

    encnew = None
    if ohe is not None:
        tmpb = ohe.transform(Xdf)
        tmpb = pd.DataFrame(tmpb, columns = ohe.get_feature_names_out())
    else:
        encnew = OneHotEncoder(categories = 'auto', handle_unknown='ignore', sparse = False, dtype = int)
        encnew.fit(Xdf)
        tmpb = encnew.transform(Xdf)
        tmpb = pd.DataFrame(tmpb, columns = encnew.get_feature_names_out())

    return tmpb, encnew


# *******************************
def onehotencoder(  # type: ignore[no-any-unimported]
    Xdf: pd.DataFrame,
    ohe: Optional [Any] = None,
    fields: Optional[Any] = None
) -> pd.DataFrame:

    """
    encode all categorical columns in pandas dataframe
    in case of model training: enhotencoder object is one of the outputs
    in case of prediction: enhotencoder object created in traing is needed 

    Args:
        Xdf (DataFrame): 
        ohe: in case of predition mandantory
             in case of training = None

    Returns:
        pandas DataFrame: binarized
        ohe - Objet (OnHotEncoing): in case of training 
                                    in case of prediction: None
    """

    dfnew, encnew = _onehotencoder(
        Xdf = Xdf, ohe = ohe, fields = fields
    )

    return dfnew, encnew

