"""
prediction based on sklearn-models like randomforrest
Created an Dezember 01 2022
@author: torchala 
""" 

from typing import Any, Optional
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _model_predict_proba(
    myML: Any,
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    igdf: Optional[pd.DataFrame] = None
) -> pd.DataFrame:

    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no column')
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no rows')

    if myML._estimator_type == 'classifier':
        ydf = pd.DataFrame(myML.predict_proba(Xdf),columns=myML.classes_)
    else:
        raise InvalidParameterValueException ('***  Model is not a classifier')

    if igdf is not None:
        if len(Xdf.index) != len(igdf.index):
            raise InvalidParameterValueException ('***  Xdf and igdf have different number of rows')
        elif len(igdf.columns) > 0:
            ydf =  pd.DataFrame(np.column_stack((igdf,ydf)),columns=igdf.columns.to_list()+ydf.columns.to_list())
    return pd.DataFrame(ydf)

# *******************************
def model_predict_proba(
    myML: Any,
    Xdf: pd.DataFrame,      # dataframe of Features for prediction
    igdf: Optional[pd.DataFrame] = None
) -> pd.DataFrame:

    """ 
        prediction of the probabititys of the classes.
        In case igdf is given. the id and geometry column will be zipped to the prediction-result
    Args:
        myML: existing classifier model to use for the prediction (random rorest  classifier, mlp classifier,... )
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        igdf (Pandas dataframe or numpy array ("array-like"), optional): columns of ids and geoemtries
    Returns:
        pandas Dataframe containg the prediction 
        for Multiclass prediction: pandas dataframe containing predicted probability values 
    """
    ydf = _model_predict_proba(
        myML=myML,
        Xdf=Xdf,
        igdf=igdf
        )

    return ydf

