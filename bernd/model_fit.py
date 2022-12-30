"""
training of a ML - model
Created an December 10 2022
@author: torchala 
""" 
#### Stand:  fast fertig
#        - Tests vor allem mit ESRI-Grids, ob die nodata-Werte korret erkannt wrden
#        - Multitarget einführen (ydf.shape.__len__()) und testen. 
#        - Weitere Tests für andere Anwendunegn (nicht Raster, sondern shape, csv, ESRI-feature-classes..)
#        - Tests mit sample_weights als 3. Argument in fit()
#             # ,sample_weight   # sample_weight: array-like of shape (n_samples,), default=None
#                                    # <= 0:   Sample wird ignoriert
#        - Tests. mit weiteren Argumenten des Sklaern-Moduls

from typing import Optional, Any
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_fit(  # type: ignore[no-any-unimported]
   myML: Any,
   Xdf: pd.DataFrame,      # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None
):

   # wenn ydf nicht angegeben ist: aus Xdf trennen
   if ydf is None:
      if fields is None:
         raise InvalidParameterValueException ('***  target and target-field are None: ') 
      else:
         name = {i for i in fields if fields[i]=="t"}
         ydf = Xdf[list(name)]
         Xdf.drop(name, axis=1, inplace=True)
   
   ty = ydf
   if len(ydf.shape) > 1: 
        if ydf.shape[1] == 1:
              ty = np.ravel(ydf)
   myML.fit(Xdf, ty)
   
   return myML

# *******************************
def model_fit(  # type: ignore[no-any-unimported]
   myML: Any, 
   Xdf: pd.DataFrame,      # dataframe of Features for training
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None
):

    """ training of a ML model

    Args:
    - myMl before trained Model (random rorest  classifier, random forest regressor,... )
    - Xdf Pandas dataframe or numpy array ("array-like") of features (columns) and samples (rows)
    - ydf Pandas dataframe or numpy array ("array-like") of target valus(columns) and samples (rows) (same number as Xdf)
        If ydf is = None, target column is included in Xdf. In this case fields should not be None
    - fields dictionary of the fieldnames and type of fields. A field type 't' is needed 
   
    Returns:
        fited ML model
    """

    myML = _model_fit(
      myML = myML,
      Xdf = Xdf, 
      ydf = ydf, 
      fields = fields
    )

    return myML
