
from typing import Optional, Any
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from copy import deepcopy
#from osgeo import ogr
from eis_toolkit.exceptions import InvalidParameterValueException, FileReadWriteError, MissingFileOrPath

# *******************************
def _export_featureclass(
    ydf: pd.DataFrame | gpd.GeoDataFrame,                                          # ydf has to add to dfg is given, else: ydf has to be exported
    dfg: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,
    metadata: Optional [Any] = None,
    outpath: Optional [str] = None,                 # path or geodatabase (.gdb)
    outfile: Optional [str] = None,                 # file or layername if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,            # if file, e.g. shape-file. shp
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,   # german Comma (,) and seperator (;)
    new_version: Optional[bool]  = False,
):

    def create_filename(
        path: str,
        name: str,
        extension: str,
        new_version: bool,
    ) -> str:
        if not os.path.exists(path):
            raise MissingFileOrPath('Path does not exists:' + str(path))
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
                try:
                    os.remove(os.path.abspath(filename+extension))
                except:
                    raise FileReadWriteError('Problems with ' + str(filename+extension))
        return path, name, extension

    def create_layername(
        path: str,
        name: str,
        #new_version: bool,
    ) -> str:
        filenum = 1
        try:
            layers = fiona.listlayers(path)
        except:
            raise FileReadWriteError('Problems with ' + str(path))
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
                    dfnew = pd.concat([dfnew, pd.DataFrame(empty)], axis=0, ignore_index=True)
                else:
                    dfnew = pd.concat([dfnew, pd.DataFrame(ydf.iloc[v]).T], axis=0, ignore_index=True)
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
                nfield = nfield + str(fieldnum)  
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
            path, file = create_layername(outpath, outfile)
            out.columns = out.columns.astype(str)
            try:
                out.to_file(path, driver='GPKG', layer=file)
            except:
                raise FileReadWriteError('Problem with writing geopackage '+str(path)+'; and layer '+str(file))
            # elif outpath.endswith('.gdb'):              # path is file geodatabase, file s layer
            #     out.to_file(path,driver='FileGDB',layer=file)
        else:
            path, file, extension = create_filename(outpath, outfile, outextension, new_version)
            filename = os.path.join(path,file)
            if outextension == "csv":
                try:
                    out.to_csv(filename+extension, header=True, sep=separator, decimal=decimal)
                except:
                    raise FileReadWriteError('Problem with writing csv-file')
            elif outextension == "shp":
                try:
                    out.to_file(filename+extension)
                except:
                    raise FileReadWriteError('Problem with writing shp-file')
            elif outextension == "geojson":
                try:
                    out.to_file(filename+extension, driver = 'GeoJSON')
                except:
                    raise FileReadWriteError('Problem with writing geojson')
            else:
                raise MissingFileOrPath('No data output. Wrong extension of the output-file') 
    return out

# *******************************
def export_featureclass(
    ydf: pd.DataFrame | gpd.GeoDataFrame,                       # ydf has to add to XDF
    dfg: Optional [gpd.GeoDataFrame | pd.DataFrame] = None,     # ydf has to be ecxported
    metadata: Optional [Any] = None,
    outpath: Optional [str] = None,
    outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,   # german Comma (,) and seperator: ;
    new_version = False,
):

    """ 
        Add the prediction result column to the existing geopandas (or pandas) dataframe (if exists)
        and saved the (geo)dataframe optionaly to a feature class file (like .shp, layer of a geopackage ...) or to a csv (text) file.
        If the file alredy exists, then the old file will be optionaly deleteted or a new file will be named numbered with (file-name)1, 2, ... 
        If outpath is a geopackage then outfile is the name of the new layer. 
        If the layer alredy exists, the new layer will be named with (layer-name)1, 2, ... 
        If the column name already exists, the new column will be named as result1, result2,... etc.
        If outfile == None: No file will be stored (the only result is a new dataframe)
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
             "True"-cells in nanmask lead to nodata-cells in ydf before added to dfg.
        If no dfg is given: 
            No nadatamask is needed.
            If ydf has a geometry column, ydf may be stored as a GIS-Layer if the extension of the file name is .shp, geojson or gpkg (no csv)
    Args:
        - ydf (pandas DataFrame): Result of prediction,
        - dfg (pandas DataFrame or GeoDataFrame): Is the primary feature table - input for prediction process (is to add with ydf)
        - metadata: In case of a geodataframe: crs (coordinate reference system)
        - outpath (string, optional): Path or name of a geopackage for the output-file or -layer
        - outfile (string, optional): Name of file or layer of the output
        - outextension (string, optional): Name of the file extension (like shp, .shp, geojson, .geojson, .gpkg or gpkg as well as csv or .csv).
        - nanmask (pandas DataFrame, optional): Marked rows witch are be droped because of nodata in the prediction input. 
        - decimalpoint_german: (bool, optional,default False): If True german decimal comma (,) and separator (;) will be used for output csv (else decimal . and separator ,)
        - new_version: = True: If the output file exists, then a new filename will be generated with the next number. 
                       = False:  The existing file will be deleted.
    Returns:
        - gpd.GeoDataFrame or pd.DataFrame (csv)
        - optionaly: stored file (csv, shp or geojson or feature class layer for a geopackage)
    """

    # Argument evaluation
    fl = []
    if not (isinstance(ydf, pd.DataFrame)):
        fl.append('ydf is not a DataFrame')
        #raise InvalidParameterValueException ('***  export_featureclass: ydf is not a DataFrame')
    if not (isinstance(dfg, (pd.DataFrame,gpd.GeoDataFrame)) or (dfg is None)):
        fl.append('dfg is not in instance of one of (pd.DataFrame,gpd.GeoDataFrame( or is not None)')
        #raise InvalidParameterValueException ('***  dfg is not in instance of one of (pd.DataFrame,gpd.GeoDataFrame,None)')
    # if not (isinstance(metadata,dict)  or (metadata is None)):
    #     fl.append('metadata is not dict or None')
    if not (isinstance(decimalpoint_german, (bool)) and isinstance(new_version,(bool))):
        fl.append('decimalpoint_german or new_version is not boolen or None')
        #raise InvalidParameterValueException ('***  decimalpoint_german or new_version is not boolen')        
    if not (isinstance(nanmask, pd.DataFrame)):
        if nanmask is not None:
            #raise InvalidParameterValueException ('*** nanmask is not an pd.DataFrame')
            fl.append('nanmask is not an pd.DataFrame or None')
    if not ((isinstance(outpath, str) or (outpath is None)) and 
        (isinstance(outfile, str) or (outfile is None)) and 
        (isinstance(outextension, str) or (outextension is None))):
        #raise InvalidParameterValueException ('***  outpath, outfile or outextension is not str (or None)')
        fl.append('outpath, outfile or outextension is not str or None')
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])
    if dfg is not None and nanmask is None:
        if ydf.shape[0] != dfg.shape[0]:          # rows (.index)
            raise InvalidParameterValueException ('ydf and dfg have not the same number of rows')
    if dfg is not None and nanmask is not None:
        if dfg.shape[0] != nanmask.shape[0]:          # rows (.index)
            raise InvalidParameterValueException ('dfg and nanamask have not the same number of rows') 
    if dfg is None and nanmask is not None:
        raise InvalidParameterValueException ('nanmask is not needed, because dfg ist None')

    return _export_featureclass(
        ydf = ydf,
        dfg = dfg,
        metadata = metadata,
        outpath = outpath,
        outfile = outfile,
        outextension = outextension,
        nanmask  = nanmask,
        decimalpoint_german = decimalpoint_german,
        new_version = new_version,
    )

