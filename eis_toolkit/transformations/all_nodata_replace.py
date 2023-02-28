
from typing import Tuple, Optional, Literal
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal['replace','mean','median','n_neighbors','most_frequent']
def _all_nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement: Optional[int] = 0,
    n_neighbors: Optional[int] = 2,
) -> Tuple[pd.DataFrame]:   #,pd.DataFrame]:     #  2th df: new target column

    # Argument evaluation
    fl = []
    if not (isinstance(df,pd.DataFrame)):
        fl.append('argument df is not a DataFrame')
    if rtype is not None:
        if not (rtype in ['replace','mean','median','n_neighbors','most_frequent']):
            fl.append('argument rtype is not in (replace,mean,median,n_neighbors,most_frequent)')
    if rtype in ['replace']:
        if not ((replacement is None) or isinstance(replacement,int)):
            fl.append('argument replacement is not integer and not None')
    if rtype in ['n_neighbors']:
        if not ((n_neighbors is None) or isinstance(n_neighbors,int)):
            fl.append('argument n_neighbors is not integer and not None')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_nodata_replace: ' + fl[0])

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
    #     raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no column')
    # if len(df.index) == 0:
    #     raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no rows')
        #df = None
    #else:
        if rtype == 'replace':
            df.fillna(replacement,inplace=True)    #df.replace(np.nan,replacement)
        elif rtype == 'n_neighbors':
            from sklearn.impute import KNNImputer
            im = KNNImputer(n_neighbors=n_neighbors,weights="uniform")
            #df = im.fit_transform(df)  #im.fit(df)
            df = pd.DataFrame(im.fit_transform(df),columns=df.columns)
        elif rtype in ['median','mean','most_frequent']:   # most_frequenty and median for categories
            from sklearn.impute import SimpleImputer
            im = SimpleImputer(missing_values=np.nan,strategy=rtype)
            df = pd.DataFrame(im.fit_transform(df),columns=df.columns) # out: dataframe
        # elif rtype == 'most_frequent':
        #     from sklearn.impute import SimpleImputer
        else:    
            raise InvalidParameterValueException ('***  function all_nodata_remove: nodata replacement not known: ' + rtype) 
    return df

#    return intern_replace(df)

# *******************************
MODE = Literal['replace','mean','median','n_neighbors','most_frequent']
def all_nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement: Optional[int | str] = 0,
    n_neighbors: Optional[int] = 2,
) -> Tuple[pd.DataFrame]:       #,pd.DataFrame]:     #  2. df new target column

    """
        replaces nodata values 
        nodata_replace shoud be used after separation (for each DataFrame separately) and bevor unification
        if nodata_replace will be used after enhotencoding: nodata values effect a nan-class for each categorical field with nodat-Vlues
    Args:
        df (Pandas DataFrame)
        type (str): 
            'replace' each nodata valu with "replacement",  does not work for string categoriesed columns!!
            'medium' replace a nodatavalue with th medium of all vlues of the feature
            'n_neighbors' replacement calculated with knieghbar - algoithm
            'most_frequent' good vor categorical columns 
        replacement (int, default = 0): value for replacemant if type is 'replace'
        n_neighbors (int, default = 2): number of neigbors if type is 'n_neighbors'

    Returns:
        - pandas DataFrame: dataframe without nodata values
    """

    df =  _all_nodata_replace(
        df = df,
        rtype = rtype,
        replacement = replacement,
        n_neighbors = n_neighbors,
    )
    return df  #, target


