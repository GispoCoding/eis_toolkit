
from typing import Optional, Any
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from copy import deepcopy
#from osgeo import ogr
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _all_export_featureclass(
    ydf: pd.DataFrame | gpd.GeoDataFrame,                                          # ydf has to add to dfg is given, else: ydf has to be exported
    dfg: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,
    metadata: Optional [Any] = None,
    outpath: Optional [str] = None,                 # path or geodatabase (.gdb)
    outfile: Optional [str] = None,                 # file or layername if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,            # if file, e.g. shape-file. shp
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,   # german Comma (,) and seperator (;)
    new_version: Optional[bool]  = False,
    #csv: Optional[bool] = False,                    # to csv-file, not fetureclass
):

    def create_filename(
        path: str,
        name: str,
        extension: str,
        new_version: bool,
    ) -> str:
        filenum = 1
        filename = os.path.join(path,name)
        if len(extension) > 0:
            if extension[0] != '.':
                extension = '.'+extension
        if (os.path.exists(os.path.abspath(filename))) or (os.path.exists(os.path.abspath(filename+extension))):
            if new_version:     # next file number
                while (os.path.exists(os.path.abspath(filename+str(filenum)))) or (os.path.exists(os.path.abspath(filename+str(filenum)+extension))):
                    filenum += 1
                return path, name+str(filenum), extension   #open(filename+str(filenum)+extension,'w')
            else:               # file will be deletet
                os.remove(os.path.abspath(filename+extension))
        return path, name, extension

    def create_layername(
        path: str,
        name: str,
        #new_version: bool,
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

    # Main program
    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    if dfg is not None and metadata is not None:
        if isinstance(dfg,gpd.GeoDataFrame):
            if metadata is not None:
                dfg.set_crs = metadata

    if dfg is None or len(ydf.columns) > 1:     # ydf to export:   ydf more then 1 column or not to add to dfg
        if nanmask is None:   # 
            #out = gpd.GeoDataFrame(ydf)  # crs = ??
            out = ydf
        else:
            # assemple a list out of the input dataframe ydf (one column???) and the nodatamask-True-values: NaN
            v = 0
            dfnew = deepcopy(ydf[0:0])
            empty = deepcopy(ydf[:1])
            empty.iloc[:] = np.nan
            #lst = []
            for cel in nanmask.iloc[:,0]:
                if cel == True:
                    dfnew = pd.concat([dfnew,pd.DataFrame(empty)],axis=0,ignore_index=True)
                else:
                    dfnew = pd.concat([dfnew,pd.DataFrame(ydf.iloc[v]).T],axis=0,ignore_index=True)
                    v += 1
            #out = gpd.GeoDataFrame(dfnew)  # crs = ??dfnew 
            out = dfnew
    else:                   # if ydf with one column (result) and to add to dfg
        out = dfg
        # next result field
        nfield = 'result'
        fieldnum = 1
        if nfield in dfg.columns:
            fieldnum = 1
            while nfield in out.columns:
                nfield + str(fieldnum)  
                fieldnum += 1

        if nanmask is None:   # 
                out[nfield] = ydf.to_numpy()
        else: # if nodata-removement was made:
            # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
            v = 0
            lst = []
            for cel in nanmask.iloc[:,0]:
                if cel == True:
                    lst.append(np.NaN)
                else:
                    lst.append(ydf.iloc[v,0])     # .values.tolist())
                    v += 1
            out[nfield] = lst
            # append as result-column
        
    # save dataframe to file or to layer in a geopacke
    if outfile is not None:
        if outextension is None:
            outextension = ''
        elif outextension[0] == '.':
            outextension = outextension[1:]
        if outpath is None:
            outpath = ''
        # geopackeges or geodatabases
        if outpath.endswith('.gpkg'):              # path is geopackage, file ist layer
            path,file = create_layername(outpath,outfile)
            out.columns = out.columns.astype(str)
            out.to_file(path,driver='GPKG',layer=file)
            # elif outpath.endswith('.gdb'):              # path is file geodatabase, file s layer
            #     out.to_file(path,driver='FileGDB',layer=file)
        else:
            path,file,extension = create_filename(outpath,outfile,outextension,new_version)
            filename = os.path.join(path,file)
            if outextension == "csv":
                out.to_csv(filename+extension,header=True,sep=separator,decimal=decimal)
            elif outextension == "shp":
                out.to_file(filename+extension)
            elif outextension == "geojson":
                out.to_file(filename+extension,driver = 'GeoJSON')
            else:
                raise InvalidParameterValueException ('***  function all_xport_featureclass: No data output. Wrong extension of the output-file')  
    return out

# *******************************
def all_export_featureclass(
    ydf: pd.DataFrame | gpd.GeoDataFrame,                       # ydf has to add to XDF
    dfg: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,     # ydf has to be ecxported
    metadata: Optional [Any] = None,
    outpath: Optional [str] = None,
    outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,   # german Comma (,) and seperator: ;
    new_version = False,
    #csv: Optional[bool] = False                    # to csv-file, not fetureclass
):

    """ 
        Add the result column to the existing geopandas (or pandas) dataframe (if exists)
        and saved the (geo)dataframe optionaly to a feature class file (like .shp) or to a csv (text) file.
        If the file alredy exists, then the old file will be optionaly delteted or the new file will be named with (file-name)1, 2, ... 
        If outpath is a geopackage then outfile is the name of the new layer. 
        If the layer alredy exists, the new layer will be named with (layer-name)1, 2, ... 
        If the column name already exists, the new column will be named as result1, result2,... etc.
        If outfile == None: No file will be stored.
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
        "True"-cells in nanmask lead to nodata-cells in ydf. So ydf may be added to dfg.
        If no dfg is given: 
            no nadatamask is needed.
            if ydf has a geometry columns, it may be stored as a GIS-Layer if tge extension of the file name is .shp, geojson or gpkg
    Args:
        - ydf (pandas DataFrame): is result of prediction,
        - dfg (pandas DataFrame or GeoDataFrame): is the primary feature table - input for prediction process (to add with ydf)
        - metadata:   in case of a geodataframe: crs
        - outpath (string, optional): Path or geodatapacke of the output-file
        - outfile (string, optional): Name of file or layer of the output
        - outextension (string, optional): Name of the file extension (like .shp, geojson or gpkg)
        - nodata (pandas DataFrame, optional): marked rows witch are be droped because of nodata in the prediction input. 
        - decimalpoint_german: (bool, optional): default False, german decimal comma (,) and separator (;)
        - new_version: = True: is the output file exists, then a new filename will be generated with the next number. 
                       = False:  the existing file will be deleted 
    Returns:
        gpd.GeoDataFrame or pd.DataFrame 
        optionaly stored as a file (csv, shp or geojson or feature class layer for a geopackage)
    """

    # Argument evaluation
    fl = []
    if not (isinstance(ydf,pd.DataFrame)):
        fl.append('ydf is not a DataFrame')
        #raise InvalidParameterValueException ('***  all_export_featureclass: ydf is not a DataFrame')
    if not (isinstance(dfg,(pd.DataFrame,gpd.GeoDataFrame)) or (dfg is None)):
        fl.append('dfg is not in instance of one of (pd.DataFrame,gpd.GeoDataFrame( or is not None)')
        #raise InvalidParameterValueException ('***  dfg is not in instance of one of (pd.DataFrame,gpd.GeoDataFrame,None)')
    # if not (isinstance(metadata,dict)  or (metadata is None)):
    #     fl.append('metadata is not dict or None')
    if not (isinstance(decimalpoint_german,(bool)) and isinstance(new_version,(bool))):
        fl.append('decimalpoint_german or new_version is not boolen or None')
        #raise InvalidParameterValueException ('***  decimalpoint_german or new_version is not boolen')        
    if not (isinstance(nanmask,pd.DataFrame)):
        if nanmask is not None:
            #raise InvalidParameterValueException ('*** nanmask is not an pd.DataFrame')
            fl.append('nanmask is not an pd.DataFrame or None')
    if not ((isinstance(outpath,str) or (outpath is None)) and 
        (isinstance(outfile,str) or (outfile is None)) and 
        (isinstance(outextension,str) or (outextension is None))):
        #raise InvalidParameterValueException ('***  outpath, outfile or outextension is not str (or None)')
        fl.append('outpath, outfile or outextension is not str or None')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_export_featureclass: ' + fl[0])
    if dfg is not None and nanmask is None:
        if ydf.shape[0] != dfg.shape[0]:          # rows (.index)
            raise InvalidParameterValueException ('*** function all_export_featureclass:  ydf and dfg have not the same number of rows')
    if dfg is not None and nanmask is not None:
        if dfg.shape[0] != nanmask.shape[0]:          # rows (.index)
            raise InvalidParameterValueException ('*** function all_export_featureclass:  dfg and nanamask have not the same number of rows') 
    if dfg is None and nanmask is not None:
        raise InvalidParameterValueException ('*** function all_export_featureclass:  nanmask is not needed, because dfg ist None')


    out = _all_export_featureclass(
        ydf = ydf,
        dfg = dfg,
        metadata = metadata,
        outpath = outpath,
        outfile = outfile,
        outextension = outextension,
        nanmask  = nanmask,
        decimalpoint_german = decimalpoint_german,
        new_version = new_version,
        #csv = csv
    )
    return out
