
from typing import Any, Optional
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _check_prediction(
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    myML: Any    
#    myOhe: Optional[Any]  
) -> dict:

    # Fields in myML are in Xdf as well? 
    q = 0
    result = {}
    t0 = set(myML.feature_names_in_) - set(Xdf.columns)
    if t0.__len__() > 0:
        q = 1
        result['missing columns in dataframe'] = t0
    t0 = set(Xdf.columns) - set(myML.feature_names_in_)
    if t0.__len__() > 0:
        q = 1
        result['missing columns in model'] = t0
    if q == 0:
        for i in range(Xdf.columns.__len__()):
            if Xdf.columns[i] != myML.feature_names_in_[i]:
                q += 1
        if q > 0: 
            result['order of the fields of dataframe and model are different'] = ''
    if q == 0:
         result = None
    return result

# *******************************
def check_prediction(
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    myML: Any    
    #myOhe: Optional[Any]  
) -> dict:

    """ 
    Check whether the fields of the dataframe are the same (number, names and ordert) as the fields in the model 

    Args:
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        myML: existing model to use for the prediction (random rorest  classifier, random forest regressor,... )

    Returns:
        Dictionary of the reult of this check
    """
    #        myOhe: existing object for OneHotEncoding of categorised fields

    return _check_prediction(Xdf, myML) #, myOhe)
