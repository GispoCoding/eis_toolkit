
from typing import Tuple, Optional, Literal
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal['replace','mean','median','n_neighbors','most_frequent']
def _nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement_number: Optional[int | float] = 0, # int
    replacement_string: Optional[str] = 'NaN', 
    n_neighbors: Optional[int] = 2,
) -> Tuple[pd.DataFrame]:   #,pd.DataFrame]:     #  2th df: new target column

    # datatype to float32
    df.loc[:,df.dtypes=='float64'] = df.loc[:,df.dtypes=='float64'].astype('float32')
    df.loc[:,df.dtypes=='int64'] = df.loc[:,df.dtypes=='int64'].astype('int32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # for col in df.columns:
    #     if not is_numeric_dtype(df[col]):     #df[col].dtype != np.number:          # # empty strings cells will get numpy.nan
    # for empty String: 
    df.replace(r'^\s*$', np.nan,regex=True,inplace=True)   # or: df.replace(r'\s+', np.nan, regex=True)
        #else:
        #     df = df.astype('float32')
        #     df.replace([np.inf, -np.inf], np.nan, inplace=True)                                   

    if not (len(df.columns) == 0 or len(df.index) == 0):
    #     raise InvalidParameterValueException ('***  function nodata_remove: DataFrame has no column')
    # if len(df.index) == 0:
    #     raise InvalidParameterValueException ('***  function nodata_remove: DataFrame has no rows')
        #df = None
    #else:
        if rtype == 'replace':               # different between: numbr and string
            for cl in df.columns:
                if df[cl].dtype == 'O':
                    df[cl].fillna(replacement_string,inplace=True)
                else:
                    df[cl].fillna(replacement_number,inplace=True)    #df.replace(np.nan,replacement)
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
            raise InvalidParameterValueException ('***  function nodata_remove: nodata replacement not known: ' + rtype) 
    return df

#    return intern_replace(df)

# *******************************
MODE = Literal['replace','mean','median','n_neighbors','most_frequent']
def nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = 'replace',
    replacement_number: Optional[int | float] = 0, # int
    replacement_string: Optional[str] = 'NaN', 
    n_neighbors: Optional[int] = 2,
) -> Tuple[pd.DataFrame]:       #,pd.DataFrame]:     #  2. df new target column

    """
        Replaces nodata values.
        nodata_replace.py shoud be used after separation.py (for each DataFrame separately) and befor unification.py
        There is no need to replace nan values in catagoriesed columns because nhotencoding creats a nan-class.
    Args:
        - df (Pandas DataFrame)
        - type (str): 
            - 'replace': Replace each nodata valu with "replacement" (see below).  Does not work for string categoriesed columns!!
            - 'medium': Replace a nodatavalue with medium of all values of the feature.
            - 'n_neighbors': Replacement calculated with k_neighbar-algorithm (see argument n_neighbors)
            - 'most_frequent': Its's suitable for categorical columns.
        replacement_number (int or float, default = 0): Value for replacement for number columns if type is 'replace'.
        replacement_string (str, default = 'NaN'): Value for replacemant for string columns if type is 'replace'. 
        n_neighbors (int, default = 2): number of neigbors if type is 'n_neighbors'

    Returns:
        - pandas DataFrame: dataframe without nodata values but with the same number of raws.
    """

   # Argument evaluation
    fl = []
    if not (isinstance(df,pd.DataFrame)):
        fl.append('argument df is not a DataFrame')
    if not (isinstance(rtype,str) or (rtype is None)):
        fl.append('argument df is not a DataFrame')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function nodata_replace: ' + fl[0])
    
    if rtype is not None:
        if not (rtype in ['replace','mean','median','n_neighbors','most_frequent']):
            fl.append('argument rtype is not in (replace,mean,median,n_neighbors,most_frequent)')
    if rtype in ['replace']:
        if not ((replacement_number is None) or isinstance(replacement_number,int) or isinstance(replacement_number,float)):
            fl.append('argument replacement is not integer, float and not None')
        if not ((replacement_string is None) or isinstance(replacement_string,str)):
            fl.append('argument replacement is not string and not None')
    if rtype in ['n_neighbors']:
        if not ((n_neighbors is None) or isinstance(n_neighbors,int)):
            fl.append('argument n_neighbors is not integer and not None')


    df =  _nodata_replace(
        df = df,
        rtype = rtype,
        replacement_number = replacement_number,
        replacement_string = replacement_string,
        n_neighbors = n_neighbors,
    )
    return df  #, target


