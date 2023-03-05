
from typing import Optional, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder #, LabelEncoder  
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _onehotencoder(
    df: pd.DataFrame,             # oder: Dateiname (dann hier erst einesen)
    ohe: Optional[Any] = None
    #fields: Optional[Any] = None
) -> pd.DataFrame:

    encnew = None
    if len(df.columns) == 0 and len(df.index) == 0:
        tmpb = None
    else:
        if ohe is not None:
            tmpb = ohe.transform(df)
            tmpb = pd.DataFrame(tmpb,columns=ohe.get_feature_names_out())  #([].append(col)))
        else:
            encnew = OneHotEncoder(categories='auto',handle_unknown='ignore',sparse = False,dtype = int)
            #encnew = OneHotEncoder(categories=col_c,handle_unknown='ignore',sparse=False,dtype = int)
            encnew.fit(df)
            tmpb = encnew.transform(df)
            tmpb = pd.DataFrame(tmpb,columns=encnew.get_feature_names_out())  #([].append(col)))

    return tmpb,encnew


# *******************************
def onehotencoder(  # type: ignore[no-any-unimported]
    df: pd.DataFrame,
    ohe: Optional [Any] = None
) -> pd.DataFrame:

    """
        Encodes all categorical columns in a pandas dataframe.
        In case of model training: onehotencoder object is one of the outputs.
        In case of prediction: onehotencoder object created in traing is needed (input argument).
    Args:
        - df (DataFrame): 
        - ohe: in case of predition mandantory
             in case of training = None

    Returns:
        pandas DataFrame: binarized
        ohe - Objet (OnHotEncoing): in case of training 
                                    in case of prediction: None
    """
    
    # Argument evaluation
    fl = []
    if not (isinstance(df,pd.DataFrame)):
        fl.append('argument df is not a DataFrame')
    t = ohe.__class__.__name__
    if not (t in ('OneHotEncoder') or ohe is None):
        fl.append('argument ohe ist not an instance of one of OneHotEncoder')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function onhotencoder: ' + fl[0])

    # if len(Xdf.columns) == 0:
    #     raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no column')
    # if len(Xdf.index) == 0:
    #     raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no rows')


    dfnew,encnew = _onehotencoder(
        df = df, ohe = ohe        #, fields = fields
    )

    return dfnew,encnew

