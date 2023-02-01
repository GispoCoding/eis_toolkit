
from typing import Tuple, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype
import geopandas as gpd
#from os.path import exists
from copy import deepcopy
#from german_to_english_decimal import *
from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.conversions.raster_to_pandas import *

#from pathlib import Path
import geopandas as gpd

# *******************************
def _import_featureclass(
    fields: dict,
    df: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,             # if None:   geofile is needed (geopandas or panda (csv))
    file: Optional [str] = None,        # csv, shp, geojson or file geodatabase, geopackage
    layer: Optional [str] = None,       # if geodatabase/geopackage: layer
    #type: Optional [str] = None,        # if shape: None, if filegeodatabase: 'FileGDB'... 
    decimalpoint_german: Optional[bool] = False
    #csv:  Optional [bool] = False
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:

    # import fiona
    # help(fiona.open)

    # parent_dir = Path(__file__).parent
    # name_fc = parent_dir.joinpath(r'data/shps/IOCG_Deps_Prosp_Occs.shp') 
    # print (name_fc)
    # read file:
    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','
    
    # Checks
    if len(fields) == 0:
        raise InvalidParameterValueException ('***  Fields is empty')

    if df is None and file is not None:
        # if not exists(file):
        #     raise InvalidParameterValueException ('***  file does not exists: ' + str(file)) 
        if file.__str__().endswith('.csv'):
            df = pd.read_csv(file,delimiter=separator,encoding='unicode_escape',decimal=decimal) 
        else:  # shp, FilGDB, 
            df = gpd.read_file(file,layer=layer) #,driver=type)
            #dfg = gpd.read_file(name_fc,layer = layer_name, driver = 'driver')
    elif df is None and file is None:
        raise InvalidParameterValueException ('***  import_featureclass: neither featureclass nor GeoDataaFrame (geopandas) is given ') 
    urdf = deepcopy(df)

    #fc = german_to_english_decimal(fc,fields)
    #crs = fc.crs
    #dfnew = gpd.GeoDataFrame()
    #gdf = geopandas.GeoDataFrame(df, geometry=gs, crs="EPSG:4326")

    # v,b,c-fields at least 1, t no more then 1
    if (list(fields.values()).count('v') + list(fields.values()).count('c') + list(fields.values()).count('d')) < 1:
        raise InvalidParameterValueException ('***  there are no v-, c- or b-fields in fields') 
    if (list(fields.values()).count('t')) > 1:
        raise InvalidParameterValueException ('***  there are more then one t-fields in fields')  
    # all fields in df?
    tmpf = {}
    for key,val in fields.items():                             # check if fields (v,b,c,t) is subset of df.columns 
        if val in ('v','b','c','t'):
            tmpf[key] = val
    if not (set(tmpf.keys()).issubset(set(df.columns))):        # 2. list is the big one, 1. list is the subset
        l = list(set(list(tmpf.keys())) - set(df.columns))
        l = ','.join(l)
        raise InvalidParameterValueException ('***  wrong columns in dataframe (compared with Fields): ' + l)

    # at least one v,c or b-Field, no more then 1 t-field
    # lv = list(fields.values()).count('v') #,'c','d')
    # if :
    #     raise InvalidParameterValueException ('***  no field type c, b or v')
    # if: 
    #     raise InvalidParameterValueException ('***  more then one fields with type t')
    # choose all columns which are in fiels

    #   alternative code: see separation
    columns = {}
    for col in df.columns:
        tmp = df[col].to_frame(name=col)
        if col in fields:
            if fields[col] in ('v','b','c','t'):    # else: column is not a value or a category 
                if 'dfnew' not in locals():                               # take the new columns
                    #q = False
                    dfnew = tmp                # fc[col].to_frame(name=col)
                else:
                    dfnew = dfnew.join(tmp)    # add an other column
                if fields[col] in ('v','b'):
                    if not is_numeric_dtype(tmp[col].dropna()):   #tmp[col].dtype != np.number:
                        raise InvalidParameterValueException ('***  v- or b-field '+col+' is not number')
                if fields[col] in ('b'):        # check b-column is 0,1
                    if not tmp[col].dropna().isin([0,1]).all():
                        raise InvalidParameterValueException ('***  b-field '+col+' is not only 0 or 1')
                columns[col] = fields[col]      # add the column name and the column type type

    return columns,dfnew,urdf

# *******************************
def import_featureclass(
    fields: dict,
    df: Optional [gpd.GeoDataFrame] = None,
    file: Optional [str] = None,
    layer: Optional [str] = None,       # if geodatabase: layer
    #type: Optional [str] = None,        # if shape: None, if filegeodatabase: 'FileGDB'... 
    decimalpoint_german: Optional[bool] = False
    #csv: Optional [bool] = False

) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:

    """
    reading a file to pandas DataFrame or csv or use a DataFrame:
    erase all not used colmuns
    creats a new fields-dictionary, e.g. {'field1': 'c'} 
    tests whether all v-type are numeric abd b-fields contain just 0 and 1
    Args:
        fields (Dictionary): name and type of the fields {'feld1': 'c'} 
            field-typs:  
            v - values (float or int)
            c - categery (int or str)
            t - target (float or int)
            b - binery (0 or 1)
            g - geometry (not to use)
            n - not to use
            i - identificator (not to use)
        df (DataFrame): DataFrame if exists. If not file should be not None
        file (string): to imported feature class  (shp) ,geodatabase, geopackage,... or csv -file
                        df should be None
        layer (string): to imported feature class (layr) from a godatabase (filegeodatabase, geopackage,... )
        type (string, default = shape ): type of the featureclass: 'FileGDB', 
        decimalpoint_german (bool): if the csv is a german coded file with comma as decimal point and semikolon as delemiter (default: False)
        csv (bool): if the file to be read is a text file (.csv), default: False

    Returns:
        dict (dictionary): in the order and the categories of the fields in the new dataframe (wihout 'n'-,'i'- or 'g'-Fields)
        df (pandas DataFrame): ready for 
           - training (with target value), 
           - test (with target value) or 
           - prediction (without target column)
        df (pandas DataFrame): the original DataFrame from import the file 
            optional: used in case of prediction to append new column as the resul of the prediction

    """
    # dfnew,columns = _import_featureclass(
    #     fields = fields, fc = fc, geofile = geofile
    #)

    return _import_featureclass(
        fields = fields,
        df = df,
        file = file,
        layer = layer, 
        #type = type,
        decimalpoint_german = decimalpoint_german
        #csv = csv
    )
