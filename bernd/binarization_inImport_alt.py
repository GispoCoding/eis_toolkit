"""
State an Dezember 08 2022
@author: torchala 
""" 
# wird nicht gebraucht, weil alles in impoert_featureclass und import_grid (muss noch eränzt werden (clumns und category-Liste))
# 
# Stand: Binarisierung exrahiert aus import_grid und somit auch für fc nutzbar
# verbessungspotenzial / offene Fragen: 
#    - Für das mehrfache Aufrufen ein übergeordnets Modul erforderlich? (siehe unten)
#    - weitere Tests, z. B. andere Bildformate: ESRI-Grid, BigTiff...?
#    - column-names are unique? If not: extend the name: name_1 or _2... 

    ### to check if: 
    #  - import file is a grid for import i rasterio
    #  - bands is a valid number of bands
    #  - name is unique in pandas columns
    #  - height and width have the same number as the first grid
    #  - categ: integer >=0, if > 0: grid must be integer with values <= categ (import as pandas: just now)
    #   if categ >0: just one band will be uses !!!
 
    # Problem, wenn es biler waren mit Bändern? 

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder      # see www: sklearn OneHotEncod
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _binarization(
    df: pd.DataFrame,
    fields: Optional[int] = 0
) -> Tuple[pd.DataFrame,dict]:

    ### OneHotEncoder (categ) and new colums names
    q = True
    cats = {}
    columns = {}
    for col in df.columns:
        tmp = df[col].to_frame(name=col)
        if fields[col] == 'c':           # if column has type category (coded in dict fields)
            enc = OneHotEncoder (handle_unknown='ignore', sparse = False)
            enc.fit(tmp)
            tmpb = enc.transform(tmp)
            tmpb = pd.DataFrame(tmpb, columns = enc.get_feature_names_out([].append(col)))
            #cats[col] = enc.categories_[0].__len__()
            cats[col] = enc.categories_[0]   # list of the categories
        else:                           # else: column is not a category
            tmpb = fields[col]            # just the df-column
        if q:                           # if output dataframe has to be created
            out = tmpb
            q = False
        else:                           # else: outpu dataframe has to extended 
            out = out.join(tmpb)

    return out, cats

# *******************************
def binarization(  # type: ignore[no-any-unimported]
    df: pd.DataFrame,
    fields: Optional[int] = 0
) -> Tuple[pd.DataFrame,dict]:

    """
    binarization (OneHotEncoder) to categorizd columns of a pandas DataFrame.

    - If Dataframe df not exists, DataFrame will be created (one column): 
      Name is the column name, to use (! name shold be unique)
    - categ > 0: grid is a grid with 0-n categories (classes). i.g. landuse , geological category ....
    - If bands are not given, all bands of the image are used for conversion. 
      Selected bands are named based on their index 
      band_1, band_2,...,band_n. 

    Args:
        df (DataFrame): to be binarized
        fields (dictionary): column type for each column

    Returns:
        pd.DataFrame: binarized DataFrame
        categories (dictionary): [column name : list of categoerises]  #or number of categories

    """

    data_frame, categories = _binarization( df = df, 
        fields = fields
    ) 

    return data_frame, categories


