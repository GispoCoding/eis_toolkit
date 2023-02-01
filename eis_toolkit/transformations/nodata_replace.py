
from typing import Tuple, Optional, Literal
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal['replace','mean','median','k_neighbors','most_frequent']
def _nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement: Optional[int] = 0,
    n_neighbors: Optional[int] = 2
) -> Tuple[pd.DataFrame]:   #,pd.DataFrame]:     #  2th df: new target column

#    def intern_replace(df):
        # datatype to float32
    df.loc[:,df.dtypes=='float64'] = df.loc[:,df.dtypes=='float64'].astype('float32')
    df.loc[:,df.dtypes=='int64'] = df.loc[:,df.dtypes=='int64'].astype('int32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # for col in df.columns:
    #     if not is_numeric_dtype(df[col]):     #df[col].dtype != np.number:          # # empty strings cells will get numpy.nan
    df.replace(r'^\s*$', np.nan,regex=True,inplace=True)   # or: df.replace(r'\s+', np.nan, regex=True)
        #else:
        #     df = df.astype('float32')
        #     df.replace([np.inf, -np.inf], np.nan, inplace=True)                                   

    if not (len(df.columns) == 0 or len(df.index) == 0):
    #     raise InvalidParameterValueException ('***  DataFrame has no column')
    # if len(df.index) == 0:
    #     raise InvalidParameterValueException ('***  DataFrame has no rows')
        #df = None
    #else:
        if rtype == 'replace':
            df.fillna(replacement,inplace=True)    #df.replace(np.nan,replacement)
        elif rtype == 'k_neighbors':
            from sklearn.impute import KNNImputer
            im = KNNImputer(n_neighbors=n_neighbors,weights="uniform")
            #df = im.fit_transform(df)  #im.fit(df)
            df = pd.DataFrame(im.fit_transform(df),columns=df.columns)
        elif rtype in ['median','mean','most_frequent']:   # most_frequenty and median for categories
            from sklearn.impute import SimpleImputer
            im = SimpleImputer(missing_values=np.nan,strategy=rtype)
            df = pd.DataFrame(im.fit_transform(df),columns=df.columns) # out: dataframe
        elif rtype == 'most_frequent':
            from sklearn.impute import SimpleImputer
        else:    
            raise InvalidParameterValueException ('***  nodata replacement not known: ' + rtype) 
    return df

#    return intern_replace(df)

# *******************************
MODE = Literal['replace','mean','median','k_neighbors','most_frequent']
def nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement: Optional[int] = 0,
    n_neighbors: Optional[int] = 2
) -> Tuple[pd.DataFrame]:       #,pd.DataFrame]:     #  2. df new target column

    """
        replaces nodata values 
        nodata_replace shoud be used after separation (for each DataFrame separately) and bevor unification
        if nodata_replace will be used after enhotencoding: nodata values effect a nan-class for each categorical field with nodat-Vlues
    Args:
        df (Pandas DataFrame)
        type (str): 
            'replace' each nodata valu with "replacement", 
            'medium' replace a nodatavalue with th medium of all vlues of the feature
            'k_neighbors' replacement calculated with knieghbar - algoithm
            'most_frequent' good vor categorical columns 

    Returns:
        - pandas DataFrame: dataframe without nodata values
    """

    df =  _nodata_replace(
        df = df,
        rtype = rtype,
        replacement = replacement,
        n_neighbors = n_neighbors
    )
    return df  #, target


