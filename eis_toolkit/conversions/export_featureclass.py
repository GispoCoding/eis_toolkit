
from typing import Optional
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from osgeo import ogr
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _export_featureclass(
    dfg: gpd.GeoDataFrame | pd.DataFrame,  # ydf has to add to XDF
    ydf: pd.DataFrame,
    outpath: Optional [str] = None,                 # path or geodatabase (.gdb)
    outfile: Optional [str] = None,                 # file or layername if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,            # if file, e.g. shape-file. shp
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False   # german Comma (,) and seperator (;)
    #csv: Optional[bool] = False                    # to csv-file, not fetureclass
) -> pd.DataFrame:

    def create_filename(
        path: str,
        name: str,
        extension: str
    ) -> str:
        filenum = 1
        filename = os.path.join(path,name)
        if len(extension) > 0:
            if extension[0] != '.':
                extension = '.'+extension
        if (os.path.exists(os.path.abspath(filename))) or (os.path.exists(os.path.abspath(filename+extension))):
            while (os.path.exists(os.path.abspath(filename+str(filenum)))) or (os.path.exists(os.path.abspath(filename+str(filenum)+extension))):
                filenum += 1
            return path, name+str(filenum), extension   #open(filename+str(filenum)+extension,'w')
        return path, name, extension

    def create_layername(
        path: str,
        name: str
    ) -> str:
        filenum = 1
        layers = fiona.listlayers(path)
        if name in layers:
            layer = name
            while layer in fiona.listlayers(path):
                layer = name+str(filenum)
                filenum += 1
            return path, layer
        return path, name

    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    out = dfg
    # next result field
    nfield = 'result'
    fieldnum = 1
    if nfield in out.columns:
        while nfield in out.columns:
            fieldnum += 1
        nfield + str(fieldnum)   

    # if nodata-removement was made:
    if nanmask is None:   # 
        out[nfield] = ydf.to_numpy()
    else:
        # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
        v = 0
        lst = []
        for cel in nanmask.iloc[:,0]:
            if cel == True:
                lst.append(np.NaN)
            else:
                lst.append(ydf.iloc[v,0])     # .values.tolist())
                v += 1
        out['result'] = lst
        # append as result-column
        
    # save dataframe to file or layer in a geopacke
    if outfile is not None:
        if outextension is None:
            outextension = ''
        if outpath is None:
            outpath = ''
        path,file,extension = create_filename(outpath,outfile,outextension)
        filename = os.path.join(path,file)
        if outextension == "csv":
            out.to_csv(filename+extension,header=True,sep=separator,decimal=decimal)
        elif outextension == "shp":
            out.to_file(filename+extension)
        elif outextension == "geojson":
            out.to_file(filename+extension,driver = 'GeoJSON')
        elif outpath.endswith('.gpkg'):
            path,file = create_layername(outpath,outfile)
        #   if outpath.endswith('.gpkg'):             # path is geopackage, file ist layer
            out.to_file(path,driver='GPKG',layer=file)
        # elif outpath.endswith('.gdb'):              # path is file geodatabase, file s layer
        #     out.to_file(path,driver='FileGDB',layer=file)
        else: 
            raise InvalidParameterValueException ('***  No data output. Wrong extension of the output-file')  
    return out

# *******************************
def export_featureclass(
    dfg: gpd.GeoDataFrame | pd.DataFrame,  # ydf has to add to XDF
    ydf: pd.DataFrame,
    outpath: Optional [str] = None,
    outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False   # german Comma (,) and seperator: ;
    #csv: Optional[bool] = False                    # to csv-file, not fetureclass
) -> pd.DataFrame:

    """ 
        Add the result column to the existing geopandas (or pandas) dataframe 
        and saved the (geo)dataframe optionaly to a feature class file (like .shp) or to a csv (text) file.
        If the file alredy exists, then new file will be named with (file-name)1, 2, ...
        If outpath is a geopackage then outfile is the name of the new layer. 
        If the layer alredy exists, the new layer will be named with (layer-name)1, 2, ...
        If the column name is given, the new column will be named as result1, result2,... etc.
        If outfile == None: No file will be stored
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
        "True"-cells in nanmask lead to nodata-cells in the output dataframe (y).
    Args:
        - dfg (pandas DataFrame or GeoDataFrame): is primary feature table - input for prediction process,
        - ydf (pandas DataFrame): is result of prediction,
        - outpath (string, optional): Path or geodatapacke of the output-file
        - outfile (string, optional): Name of file or layer of the output
        - outextension (string, optional): Name of the file extension (like .shp)
        - nodata (pandas DataFrame, optional): marked rows witch are be droped because of nodata in the prediction input. 
        - decimalpoint_german: (bool, optional): default False, german decimal comma (,) and separator (;)
    Returns:
        gpd.GeoDataFrame or pd.DataFrame 
        as stored file (csv, shp or geojson or feature class layer 
    """

    out = _export_featureclass(
        dfg = dfg,
        ydf = ydf,
        outpath = outpath,
        outfile = outfile,
        outextension = outextension,
        nanmask  = nanmask,
        decimalpoint_german = decimalpoint_german
        #csv = csv
    )
    return out
