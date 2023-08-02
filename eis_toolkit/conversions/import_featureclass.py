import pathlib
from copy import deepcopy
from os.path import exists

import geopandas as gpd
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Tuple, Union
from pandas.api.types import is_numeric_dtype

from eis_toolkit.exceptions import FileReadError, InvalidParameterValueException


# *******************************
@beartype
def _import_featureclass(
    fields: dict,
    df: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,  # if None:   geofile is needed (geopandas or panda (csv))
    file: Optional[Union[str, pathlib.PosixPath]] = None,  # csv, shp, geojson or file geodatabase, geopackage
    layer: Optional[str] = None,  # if geodatabase/geopackage: layer
    decimalpoint_german: Optional[bool] = False,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, Any]:

    # main
    if decimalpoint_german:
        decimal = ","
        separator = ";"
    else:
        decimal = "."
        separator = ","

    metadata = None
    # check
    if df is None and file is not None:
        if not exists(file):
            raise InvalidParameterValueException("File does not exists: " + str(file))
        if file.__str__().endswith(".csv"):
            try:
                df = pd.read_csv(file, delimiter=separator, encoding="unicode_escape", decimal=decimal)
            except:
                raise FileReadError("File is not readable " + str(file))
        else:  # shp, FilGDB
            try:
                df = gpd.read_file(file, layer=layer)
            except:
                raise FileReadError("File is not readable " + str(file))
            # save the coordinata reference system (crs) to metadata
            if df.crs is not None:
                metadata = df.crs
    elif df is None and file is None:
        raise InvalidParameterValueException("Neither file (featureclass, csv) nor df (DataFrame (geopandas)) is given")
    urdf = deepcopy(df)

    # v,b,c-fields at least 1, t no more then 1
    if (list(fields.values()).count("v") + list(fields.values()).count("c") + list(fields.values()).count("b")) < 1:
        raise InvalidParameterValueException("There are no v-, c- or b-fields in fields argument")
    if (list(fields.values()).count("t")) > 1:
        raise InvalidParameterValueException("There are more then one t-fields in fields argument")
    # all fields in df?
    tmpf = {}
    for key, val in fields.items():  # check if fields (v,b,c,t) is subset of df.columns
        if val in ("v", "b", "c", "t"):
            tmpf[key] = val
    if not (set(tmpf.keys()).issubset(set(df.columns))):  # 2. list is the big one, 1. list is the subset
        l = list(set(list(tmpf.keys())) - set(df.columns))
        l = ",".join(l)
        raise InvalidParameterValueException("Wrong columns in dataframe (compared with Fields): " + l)

    #   alternative code: see separation
    columns = {}
    for col in df.columns:
        tmp = df[col].to_frame(name=col)
        if col in fields:
            if fields[col] in ("v", "b", "c", "t", "i", "g"):  # else: column is not a value or a category
                if "dfnew" not in locals():  # take the new columns
                    dfnew = tmp
                else:
                    dfnew = dfnew.join(tmp)  # add an other column
                if fields[col] in ("v", "b"):
                    if not is_numeric_dtype(tmp[col].dropna()):
                        raise InvalidParameterValueException("v- or b-field " + col + " is not a number")
                if fields[col] in ("b"):  # check b-column is 0,1
                    if not tmp[col].dropna().isin([0, 1]).all():
                        raise InvalidParameterValueException("b-field " + col + " is not only 0 or 1")
                columns[col] = fields[col]  # add the column name and the column type type

    return columns, dfnew, urdf, metadata


# *******************************
@beartype
def import_featureclass(
    fields: dict,
    df: Optional[Union[gpd.GeoDataFrame , pd.DataFrame]] = None,
    file: Optional[Union[str, pathlib.PosixPath]] = None,
    layer: Optional[str] = None,  # if geodatabase: layer
    decimalpoint_german: Optional[bool] = False,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, Any]:

    """
        Reading a file to pandas DataFrame or csv or use an existing DataFrame instad of the file.
        Erase all not used colmuns in the DataFrame
        Creats a new fields-dictionary (columns), e.g. {'field1': 'c'}
        Tests whether all v-type are numeric and b-fields contain just 0 and 1
    Args:
        - fields: name and type of the fields {'feld1': 'c'}
            field-types:
            v - values (float or int)
            c - category (int or str)
            t - target (float, int or str)
            b - binery (0 or 1)
            g - geometry
            n - not to use
            i - identifier
        - df: DataFrame if exists. If not file is given df should be not None.
        - file: To imported feature class (shp ,geodatabase, geopackage,...) or csv -file.
                        If file is not None, df is not needed.
        - layer: To import feature class (layer) from a geodatabase (geodatabase, geopackage,... )
        - decimalpoint_german (bool): If the csv is a german coded file with comma as decimal point and semikolon as delemiter (default: False)

    Returns:
        - dictionary: Columns in order and with the types of the fields in the new dataframe (from imput fields but without 'n'-Fields)
        - pandas DataFrame to use for
           - model training (with target value),
           - model validation (with target value) or
           - model prediction (without target column)
        - The original pandas DataFrame from import the file
            Used in case of prediction to append a new column as the resul of the prediction
        - metadata /crs-object: Contains the coordinate reference system (crs) if exists
            optional: metadata dictionary if an geodatandas dataframe is imported.
                      in this case crs is stored in metadata['crs']
    """

    return _import_featureclass(fields=fields, df=df, file=file, layer=layer, decimalpoint_german=decimalpoint_german)
