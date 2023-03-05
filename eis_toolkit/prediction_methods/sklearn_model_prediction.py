
from typing import Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _sklearn_model_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
):

    cn = ['result']
    ydf = sklearnMl.predict(Xdf)

    if sklearnMl._estimator_type == 'classifier':
        if 'float' in ydf.dtype.__str__():      # in case of classification problem and the target is not int
            ydf = (ydf + 0.5).astype(np.uint16)
        elif 'int' in ydf.dtype.__str__():
            ydf = ydf.astype(np.uint16)

    ydf= pd.DataFrame(ydf,columns=cn)
    if igdf is not None:
        if len(ydf.index) != len(igdf.index):
            raise InvalidParameterValueException ('***  function sklearn_model_prediction: Xdf and igdf have different number of rows')
        elif len(igdf.columns) > 0:              # zipping og id and geo columns
            ydf =  pd.DataFrame(np.column_stack((igdf,ydf)),columns=igdf.columns.to_list()+ydf.columns.to_list())
            #ydf =  pd.DataFrame(zip(igdf,ydf), columns=[igdf.columns,ydf.columns])
            gm =  list({i for i in fields if fields[i] in ('g')})
            if len(gm) == 1:                      # if geometry exists DataFrame will be changed to geoDataFrame
                if gm == ['geometry']:
                    ydf = gpd.GeoDataFrame(ydf)

    return ydf #,columns=cn)

# *******************************
def sklearn_model_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,      # dataframe of Features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
):

    """ 
        Prediction based on a SKLEARN model and  of a DataFrame of samples.
        If given the result will be zipped to Id and geometry columns.
    Args:
        sklearnMl: Existing model to use for the prediction (random rorest classifier, random forest regressor, logistic regression).
        Xdf (Pandas dataframe or numpy array ("array-like")): Features (columns) and samples (raws) of samples to predict y.
        igdf (Pandas dataframe or numpy array ("array-like"), optional): Columns of ids and geoemtries.
        fields (Dictionary, optinal): If given it will be used to set the geometry for geodataframe.   
    Returns:
        pandas dataframe or geodataframe containing predicted values. (if geodataframe:  geoemtry columns is in the geodataframe)
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
        raise InvalidParameterValueException ('***  function sklearn_model_prediction: ' + fl[0])
    
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException ('***  function sklearn_model_prediction: DataFrame has no column')
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException ('***  function sklearn_model_prediction: DataFrame has no rows')
    
    ydf = _sklearn_model_prediction(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        igdf=igdf,
        fields=fields,
        )

    return ydf
