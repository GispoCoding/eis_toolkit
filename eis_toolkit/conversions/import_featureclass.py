
from typing import Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype
import geopandas as gpd
from os.path import exists
from copy import deepcopy
from eis_toolkit.exceptions import InvalidParameterValueException, FileReadWriteError
from eis_toolkit.conversions.raster_to_pandas import *
import geopandas as gpd

# *******************************
def _import_featureclass(
    fields: dict,
    df: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,             # if None:   geofile is needed (geopandas or panda (csv))
    file: Optional [str] = None,        # csv, shp, geojson or file geodatabase, geopackage
    layer: Optional [str] = None,       # if geodatabase/geopackage: layer
    decimalpoint_german: Optional[bool] = False,
):

    # main
    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    metadata = None  
    # check
    if df is None and file is not None:
        if not exists(file):
             raise InvalidParameterValueException('File does not exists: ' + str(file)) 
        if file.__str__().endswith('.csv'):
            try:
                df = pd.read_csv(file, delimiter=separator, encoding='unicode_escape', decimal=decimal) 
            except:
                raise FileReadWriteError('File is not readable '+str(file))
        else:  # shp, FilGDB,
            try:
                df = gpd.read_file(file, layer=layer) #,driver=type)
            except:
                raise FileReadWriteError('File is not readable '+str(file))
            # save the coordinata reference system (crs) to metadata
            if df.crs is not None:
                metadata = df.crs          #.to_epsg()
            #dfg = gpd.read_file(name_fc,layer = layer_name, driver = 'driver')
    elif df is None and file is None:
        raise InvalidParameterValueException ('Neither file (featureclass, csv) nor df (DataFrame (geopandas)) is given') 
    urdf = deepcopy(df)

    # v,b,c-fields at least 1, t no more then 1
    if (list(fields.values()).count('v') + list(fields.values()).count('c') + list(fields.values()).count('d')) < 1:
        raise InvalidParameterValueException ('There are no v-, c- or b-fields in fields argument') 
    if (list(fields.values()).count('t')) > 1:
        raise InvalidParameterValueException ('There are more then one t-fields in fields argument')  
    # all fields in df?
    tmpf = {}
    for key,val in fields.items():                             # check if fields (v,b,c,t) is subset of df.columns 
        if val in ('v','b','c','t'):
            tmpf[key] = val
    if not (set(tmpf.keys()).issubset(set(df.columns))):        # 2. list is the big one, 1. list is the subset
        l = list(set(list(tmpf.keys())) - set(df.columns))
        l = ','.join(l)
        raise InvalidParameterValueException ('Wrong columns in dataframe (compared with Fields): ' + l)

    # at least one v,c or b-Field, no more then 1 t-field
    # lv = list(fields.values()).count('v') #,'c','d')
    # if :
    #     raise InvalidParameterValueException ('no field type c, b or v')
    # if: 
    #     raise InvalidParameterValueException ('more then one fields with type t')
    # choose all columns which are in fiels

    #   alternative code: see separation
    columns = {}
    for col in df.columns:
        tmp = df[col].to_frame(name=col)
        if col in fields:
            if fields[col] in ('v','b','c','t','i','g'):    # else: column is not a value or a category 
                if 'dfnew' not in locals():                               # take the new columns
                    #q = False
                    dfnew = tmp                # fc[col].to_frame(name=col)
                else:
                    dfnew = dfnew.join(tmp)    # add an other column
                if fields[col] in ('v','b'):
                    if not is_numeric_dtype(tmp[col].dropna()):   #tmp[col].dtype != np.number:
                        raise InvalidParameterValueException ('v- or b-field '+col+' is not a number')
                if fields[col] in ('b'):        # check b-column is 0,1
                    if not tmp[col].dropna().isin([0,1]).all():
                        raise InvalidParameterValueException ('b-field '+col+' is not only 0 or 1')
                columns[col] = fields[col]      # add the column name and the column type type

    return columns, dfnew, urdf, metadata

# *******************************
def import_featureclass(
    fields: dict,
    df: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,
    file: Optional [str] = None,
    layer: Optional [str] = None,       # if geodatabase: layer
    decimalpoint_german: Optional[bool] = False,
):

    """
        Reading a file to pandas DataFrame or csv or use an existing DataFrame instad of the file.
        Erase all not used colmuns in the DataFrame
        Creats a new fields-dictionary (columns), e.g. {'field1': 'c'} 
        Tests whether all v-type are numeric and b-fields contain just 0 and 1
    Args:
        - fields (Dictionary): name and type of the fields {'feld1': 'c'} 
            field-types:  
            v - values (float or int)
            c - category (int or str)
            t - target (float, int or str)
            b - binery (0 or 1)
            g - geometry
            n - not to use
            i - identifier
        - df (DataFrame): DataFrame if exists. If not file is given df should be not None.
        - file (string): To imported feature class (shp ,geodatabase, geopackage,...) or csv -file.
                        If file is not None, df is not needed.
        - layer (string): To import feature class (layer) from a geodatabase (geodatabase, geopackage,... )
        - decimalpoint_german (bool): If the csv is a german coded file with comma as decimal point and semikolon as delemiter (default: False)

    Returns:
        - dict (dictionary): Columns in order and with the types of the fields in the new dataframe (from fields but without 'n'-Fields)
        - df (pandas DataFrame): To use for 
           - model training (with target value), 
           - model validation (with target value) or 
           - model prediction (without target column)
        - df (pandas DataFrame): The original DataFrame from import the file 
            optional: used in case of prediction to append a new column as the resul of the prediction
        - metadata (crs-object): Contains the coordinate reference system if exists
            optional: metadata dictionary if an geodatandas dataframe is imported. crs is in metadata['crs']
    """

    # Argument evaluation
    fl = []
    if not (isinstance(df, (gpd.GeoDataFrame,pd.DataFrame)) or df is None):
        fl.append('Argument df is not a DataFrame and is not None')
    if not ((isinstance(file.__str__(), str) or file is None) and 
        (isinstance(layer, str) or layer is None)):
        fl.append('Arguments file or layer are not str or are not None')
    if not ((isinstance(fields, dict)) or fields is None):
        fl.append('Argument fields is not a dictionary and is not None')
    if not (isinstance(decimalpoint_german, (bool)) or decimalpoint_german is None):
        fl.append('Argument decimalpoint_german are not boolen or are not None')      
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])

    # Checks
    if len(fields) == 0:
        raise InvalidParameterValueException ('Argument fields is empty')

    return _import_featureclass(
        fields = fields,
        df = df,
        file = file,
        layer = layer, 
        decimalpoint_german = decimalpoint_german
    )
