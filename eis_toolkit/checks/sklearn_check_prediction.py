
from typing import Any
import numpy as np
import pandas as pd

from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _sklearn_check_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of features for prediction
#    myOhe: Optional[Any]  
) -> pd.DataFrame:

    # evaluation arguments
    # Argument evaluation
    fl = []        
    if not (isinstance(Xdf,pd.DataFrame)):
        fl.append('argument Xdf is not a DataFrame')
        #raise InvalidParameterValueException ('***  Xdf is not a DataFrame')
    t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
    if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
        fl.append('argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
        #raise InvalidParameterValueException ('***  sklearnMl ist not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function sklearn_check_prediction: ' + fl[0])
    
    # Fields in sklearnMl are in Xdf as well? 
    t0 = set(sklearnMl.feature_names_in_) - set(Xdf.columns)
    if t0.__len__() > 0:
        raise InvalidParameterValueException ('***  function sklearn_check_prediction: missing columns in dataframe (compared with Model): ' + str(t0))
    t0 = set(Xdf.columns) - set(sklearnMl.feature_names_in_)
    if t0.__len__() > 0:
        raise InvalidParameterValueException ('***  function sklearn_check_prediction: wrong columns in dataframe  (compared with Model): ' + str(t0))
    if not (Xdf.shape[1] == Xdf.select_dtypes(include=np.number).shape[1]):
        raise InvalidParameterValueException ('***  function sklearn_check_prediction: non numeric data in the Dataframe') 
    else:
        #Xdf = Xdf[[sklearnMl.feature_names_in_]]
        Xdf = Xdf.reindex(columns=sklearnMl.feature_names_in_) # alternative
    # if q == 0:
    #     for i in range(Xdf.columns.__len__()):
    #         if Xdf.columns[i] != sklearnMl.feature_names_in_[i]:
    #             q += 1
    #     if q > 0: 
    #         result['order of the fields of dataframe and model are different'] = ''

    #      result = None
    # else:
    return Xdf

# *******************************
def sklearn_check_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of features for prediction   
    #myOhe: Optional[Any]  
) -> pd.DataFrame:

    """ 
        Check whether the fields of the dataframe are the same (number and names) as the fields in the model 
        The Columns will be ordered in the same way as in sklearnMl used
        Check_Prediction should be used just befor model_prediction, after onehotencoding
    Args:
        sklearnMl: existing model to use for the prediction (random rorest  classifier, random forest regressor,... )
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
    Returns:
        Dataframe with columns in the order of the Model sklearnMl
    """
    #        myOhe: existing object for OneHotEncoding of categorised fields

    return _sklearn_check_prediction(sklearnMl,Xdf) #, myOhe)
