
from typing import Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _sklearn_model_predict_proba(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
):

    if sklearnMl._estimator_type == 'classifier':
        ydf = pd.DataFrame(sklearnMl.predict_proba(Xdf),columns=sklearnMl.classes_)
    else:
        raise InvalidParameterValueException ('***  function sklearn_model_predict_proba:  Model is not a classifier')

    if igdf is not None:
        if len(ydf.index) != len(igdf.index):
            raise InvalidParameterValueException ('***  function sklearn_model_predict_proba:  Xdf and igdf have different number of rows')
        elif len(igdf.columns) > 0:              # zipping og id and geo columns
            ydf =  pd.DataFrame(np.column_stack((igdf,ydf)),columns=igdf.columns.to_list()+ydf.columns.to_list())
            gm =  list({i for i in fields if fields[i] in ('g')})
            if len(gm) == 1:
                if gm == ['geometry']:           # if geometry exists DataFrame will be changed to geoDataFrame
                    ydf = gpd.GeoDataFrame(ydf)
    return ydf

# *******************************
def sklearn_model_predict_proba(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of Features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
):

    """ 
        Probabititys of the Prediction of the classes.
        In case igdf is given. the id and geometry column will be zipped to the probability result.
    Args:
        - sklearnMl: Existing classifier model to use for calculation of the prediction probability (random forest classifier, logistic regression,... )
        - Xdf (Pandas dataframe or numpy array ("array-like")): Features (columns) and samples (raws) to use for calculation of the prediction probability
        - igdf (Pandas dataframe or numpy array ("array-like"), optional): Columns of identification and geoemtries of the raws
        - fields (Dictionary, optinal): If given it will be used to set the geometry for geodataframe
    Returns:
        pandas Dataframe or geoDataFram containg the prediction probability values for Multiclass prediction
    """

    # Argument evaluation
    fl = []
    if not (isinstance(Xdf,pd.DataFrame)):
        fl.append('argument Xdf is not a DataFrame')
    t = sklearnMl.__class__.__name__
    if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
        fl.append('argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
    if not (isinstance(igdf,pd.DataFrame) or (igdf is None)):
        fl.append('argument igdf is not a DataFrame and is not None')
    if not (isinstance(fields,dict) or (fields is None)):
        fl.append('argument fields is not a dictionary and is not None')            
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function sklearn_model_predict_proba:  ' + fl[0])
    
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException ('***  function sklearn_model_predict_proba: DataFrame has no column')
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException ('***  function sklearn_model_predict_proba:  DataFrame has no rows')


    ydf = _sklearn_model_predict_proba(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        igdf=igdf,
        fields = fields,
    )

    return ydf
