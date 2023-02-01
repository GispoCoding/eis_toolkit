
from typing import Optional, Any
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_fit(  # type: ignore[no-any-unimported]
   myML: Any,
   Xdf: pd.DataFrame,                       # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None
):

   if len(Xdf.columns) == 0:
      raise InvalidParameterValueException ('***  DataFrame has no column')
   if len(Xdf.index) == 0:
      raise InvalidParameterValueException ('***  DataFrame has no rows')

   # if ydf is None: to split from Xdf 
   if ydf is None:
      if fields is None:
         raise InvalidParameterValueException ('***  target and target-field are None') 
      else:
         name = {i for i in fields if fields[i]=="t"}
         ydf = Xdf[list(name)]
         Xdf.drop(name,axis=1,inplace=True)
   
   ty = ydf
   if len(ydf.shape) > 1: 
        if ydf.shape[1] == 1:
              ty = np.ravel(ydf)
   myML.fit(Xdf,ty)
   
   return myML

# *******************************
def model_fit(  # type: ignore[no-any-unimported]
   myML: Any,
   Xdf: pd.DataFrame,                       # dataframe of Features for training
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None
):

    """ 
      training of a ML model
    Args:
      - myMl: before trained Model (random rorest  classifier, random forest regressor,... )
      - Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (rows)
      - ydf (Pandas dataframe or numpy array ("array-like")): target valus(columns) and samples (rows) (same number as Xdf)
         If ydf is = None, target column is included in Xdf. In this case fields should not be None
      - fields (dictionary): the fieldnames and type of fields. A field type 't' is needed. 
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
