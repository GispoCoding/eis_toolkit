"""
replace nodata-Vales or delete samples with nodatavalues 
State an Dezember 03 2022
@author: torchala 
""" 

### Stand: fast fertig, zu verbessern und zu prüfen (tests)

from typing import Tuple, Optional, Literal
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal['replace', 'mean','median', 'k_neighbors', 'most_frequent']
def _nodata_replace(
    Xdf: pd.DataFrame,
    ydf: Optional[pd.DataFrame] = None,        # y-DataFrame (target for model traing purposes)
    rtype: Optional[MODE] = 'replace',
    replacement: Optional[int] = 0,
    n_neighbors: Optional[int] = 2
) -> Tuple[pd.DataFrame,pd.DataFrame]:     #  2. df new target column

    def intern_replace(df):
        # datatype to float32
        df = df.astype('float32')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if rtype == 'replace':
            df.fillna(replacement, inplace = True)    #df.replace(np.nan,replacement)
        elif rtype == 'k_neighbors':
            from sklearn.impute import KNNImputer
            im = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
            #df = im.fit_transform(df)  #im.fit(df)
            df = pd.DataFrame(im.fit_transform(df),columns = df.columns)
        elif rtype in ['median','mean','most_frequent']:   # most_frequenty and median for categories
            from sklearn.impute import SimpleImputer
            im = SimpleImputer(missing_values=np.nan, strategy=rtype)
            df = pd.DataFrame(im.fit_transform(df),columns = df.columns) # out: dataframe
        elif rtype == 'most_frequent':
            from sklearn.impute import SimpleImputer
        else:    
            raise InvalidParameterValueException ('***  nodata replacement not known: ' + rtype) 

        return df           # if no ydf (target) exists: thease output DataFrames are set to None

    if ydf is not None: 
        ydf = intern_replace(ydf) 

    return intern_replace(Xdf), ydf

# *******************************
MODE = Literal['replace', 'mean','median', 'k_neighbors','most_frequent']
def nodata_replace(
    Xdf: pd.DataFrame,
    ydf: Optional[pd.DataFrame] = None,
    rtype: Optional[MODE] = 'replace',  
    replacement: Optional[int] = 0,
    n_neighbors: Optional[int] = 2
) -> Tuple[pd.DataFrame,pd.DataFrame]:     #  2. df new target column

    """
    replaces nodata values 

    Args:
        Xdf (Pandas DataFrame)
        ydf  (Pandas DataFrame): is exists (for prediction)
        type (str): 
        'replace' each nodata valu with "replacement", 
        'medium' replace a nodatavalue with th medium of all vlues of the feature
        'k_neighbors' replacement calculated with knieghbar - algoithm
        'most_frequent' good vor categorical columns

    Returns:
        - pd.DataFrame: dataframe without nodata values
        - target DataFrame:  for training purposes (in case of prediction target df will be Non)
        
    """

    df,target =  _nodata_replace(                # np2: nodatamask
        Xdf = Xdf,
        ydf = ydf,
        rtype = rtype,
        replacement = replacement,
        n_neighbors = n_neighbors
    )
    return df, target


