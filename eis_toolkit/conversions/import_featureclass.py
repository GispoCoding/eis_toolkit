
from typing import Tuple
import pandas as pd
import geopandas as gpd
from eis_toolkit.exceptions import InvalidParameterValueException
from pathlib import Path
import geopandas as gpd

# *******************************
def _import_featureclass(
    fc: gpd.GeoDataFrame | pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame, dict]:
    
    q = True   # first loop
    columns = {}
    for col in fc.columns:
        tmp = fc[col].to_frame(name=col)
 
        if fields[col] not in ('n','i','g'):    # else: column is a value or a category  
            if q:                          # take the new columns
                q = False
                dfnew = tmp                # = fc[col].to_frame(name=col)
            else:
                dfnew = dfnew.join(tmp)    # add an other column

            columns[col] = fields[col]     # add the column name and the column type

    return dfnew, columns

# *******************************
def import_featureclass(
    fc: gpd.GeoDataFrame,
    fields: dict
) -> Tuple[pd.DataFrame, dict]:

    """
    imports feature class to pandas DataFrame and erase all not used colmns
    creats a new fields-dictionary, e.g. {'field1': 'c'} 

    Args:
        fc (DataFrame): to imported feature class (shp, geodatabase layer, geopackage layer,...)
        fields (Dictionary): name and type of the fields {'feld1': 'c'} 
            field-typs:  
            v - values
            c - categery
            z - target
            b - binery
            g - geometry
            n - not to use

    Returns:
        df (pandas DataFrame): ready for training (with target value), test (with target value) or prediction (without target column)
        dict (dictionary): in the order and the categories of the fields in the new dataframe (wihout 'n'-,'i'- or 'g'-Fields)
    """
    dfnew,columns = _import_featureclass(
        fc = fc, fields = fields
    )

    return dfnew ,columns

