"""
prediction based on sklearn-models like randomforrest
Created an Dezember 01 2022
@author: torchala 
""" 
#### Stand:  erste Tests ok. 
#        - Tests vor allem mit ESRI-Grids, ob die nodata-Werte korret erkannt wrden
#               offene Frage: siehe unten
#        - Multitarget einführen (ydf.shape.__len__()) und testen. Woher kommen dann die Namen der Salten?
#        - Weitere Tests für andere Anwednunegn (nicht Raster, sondern shape, csv, ESRI-feature-classes..)

from typing import Any
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _model_prediction(
    myML: Any,      
    Xdf: pd.DataFrame      # dataframe of features for prediction
) -> pd.DataFrame:

    ydf = myML.predict(Xdf)
    if myML._estimator_type == 'classifier':
        if 'int' not in ydf.dtype.__str__():      # in case of classification problem and the target is not int
            ydf = (ydf + 0.5).astype(np.uint16)
        else:
            ydf = ydf.astype(np.uint16)
    cn = ['result']

    return pd.DataFrame(ydf,columns=cn)

# *******************************
def model_prediction(
    myML: Any,      
    Xdf: pd.DataFrame      # dataframe of Features for prediction
) -> pd.DataFrame:

    """ training of a randmon forest model

    Args:
        myML: existing model to use for the prediction (random rorest  classifier, random forest regressor,... )
        Xdf: Pandas dataframe or numpy array ("array-like") of features (columns) and samples (raws)

    Returns:
        ydf - pandas dataframe containing predicted values
    """

    ydf = _model_prediction(myML, Xdf)  

    return ydf

