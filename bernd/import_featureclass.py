"""
import a featureclass to GeopndasDataFrame, incl. target column
Created an Dezember 13 2022
@author: torchala 
""" 

### zu Verzeichnis canversion
# Stand: fast fertig
# 
# offene Fragen:  
# - Benutzerdialog auf Basis der fc in qgis oder der als geopandas eingelesenen fc???? Wichtige Frage, damit das fc nicht zweimal eingelesen werden muss.

from typing import Tuple
import pandas as pd
import geopandas as gpd
from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.conversions.raster_to_pandas import *

#from pathlib import Path
import geopandas as gpd

# *******************************
def _import_featureclass(
    fc: gpd.GeoDataFrame | pd.DataFrame,             # oder: Dateiname (dann hier erst einlesen) 
    fields: dict
) -> Tuple[pd.DataFrame, dict]:

    # import fiona
    # help(fiona.open)

    # parent_dir = Path(__file__).parent
    # name_fc = parent_dir.joinpath(r'data/shps/IOCG_Deps_Prosp_Occs.shp') 
    # print (name_fc)

    #dfg = gpd.read_file(fc)
    #crs = fc.crs
    #dfnew = gpd.GeoDataFrame()

    #gdf = geopandas.GeoDataFrame(df, geometry=gs, crs="EPSG:4326")
    
    q = True   # first loop
    columns = {}
    for col in fc.columns:
        tmp = fc[col].to_frame(name=col)
 
        if fields[col] not in ('n','i','g'):    # else: column is not a value or a category  
            if q:                                # take the new columns
                q = False
                dfnew = tmp                # fc[col].to_frame(name=col)
            else:
                dfnew = dfnew.join(tmp)    # add an other column

            columns[col] = fields[col]      # add the column name and the column type type

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

