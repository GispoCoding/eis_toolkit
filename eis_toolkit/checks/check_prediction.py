
from typing import Any, Optional
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _check_prediction(
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    myML: Any
#    myOhe: Optional[Any]  
) -> pd.DataFrame:

    # Fields in myML are in Xdf as well? 

    t0 = set(myML.feature_names_in_) - set(Xdf.columns)
    if t0.__len__() > 0:
        raise InvalidParameterValueException ('***  missing columns in dataframe (compared with Model): ' + str(t0))
    t0 = set(Xdf.columns) - set(myML.feature_names_in_)
    if t0.__len__() > 0:
        raise InvalidParameterValueException ('***  wrong columns in dataframe  (compared with Model): ' + str(t0))
    if not (Xdf.shape[1] == Xdf.select_dtypes(include=np.number).shape[1]):
        raise InvalidParameterValueException ('***  non numeric data in the Dataframe') 
    else:
        #Xdf = Xdf[[myML.feature_names_in_]]
        Xdf = Xdf.reindex(columns=myML.feature_names_in_) # alternative
    # if q == 0:
    #     for i in range(Xdf.columns.__len__()):
    #         if Xdf.columns[i] != myML.feature_names_in_[i]:
    #             q += 1
    #     if q > 0: 
    #         result['order of the fields of dataframe and model are different'] = ''

    #      result = None
    # else:
    return Xdf

# *******************************
def check_prediction(
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    myML: Any    
    #myOhe: Optional[Any]  
) -> pd.DataFrame:

    """ 
        Check whether the fields of the dataframe are the same (number und names) as the fields in the model 
        The Columns will be ordered in the same way as in myML used
        Check_Prediction should be used just befor model_prediction, after onehotencoding
    Args:
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        myML: existing model to use for the prediction (random rorest  classifier, random forest regressor,... )
    Returns:
        Dataframe with columns in the order of tne Model myML
    """
    #        myOhe: existing object for OneHotEncoding of categorised fields

    return _check_prediction(Xdf,myML) #, myOhe)
