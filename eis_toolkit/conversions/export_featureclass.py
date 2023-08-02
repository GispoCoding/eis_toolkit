import os
import pathlib
from copy import deepcopy

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Union
from geopandas import GeoDataFrame

from eis_toolkit.exceptions import FileWriteError, InvalidParameterValueException


# *******************************
@beartype
def _export_featureclass(
    ydf: Union[pd.DataFrame, gpd.GeoDataFrame],  # ydf has to add to dfg is given, else: ydf has to be exported
    dfg: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    igdf: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    metadata: Optional[Any] = None,
    outpath: Optional[Union[str, pathlib.PosixPath]] = None,  # path or geodatabase (.gdb)
    outfile: Optional[str] = None,         # file or layername if not given: pd.DataFrame will given back
    outextension: Optional[str] = None,           # if file, e.g. shape-file. shp
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,  # german Comma (,) and seperator (;)
    new_version: Optional[bool] = False,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    def create_filename(
        path: str,
        name: str,
        extension: str,
        new_version: bool,
    ):
        if not os.path.exists(path) and path != "":
            raise InvalidParameterValueException("Parameter outpath is not an existing path")
        # the path may be emty.in this case the filename contains the path
        filename = os.path.join(path, name)
        if len(extension) > 0:
            if extension[0] != ".":
                extension = "." + extension
        # in case the tputfile exists, a new filename will be created or the existing file will be deleted
        filenum = 1
        if os.path.exists(os.path.abspath(filename + extension)):
            if new_version:  # next file number
                while os.path.exists(os.path.abspath(filename + str(filenum) + extension)):
                    filenum += 1
                return path, name + str(filenum), extension  # open(filename+str(filenum)+extension,'w')
            # file will be deletet
            os.remove(os.path.abspath(filename + extension))
        return path, name, extension

    def create_layername(
        path: str,
        name: str,
    ):
        filenum = 1
        layers = fiona.listlayers(path)
        # if the layer exists, a new name will be created or the layer will be deleted
        if name in layers:
            layer = name
            while layer in fiona.listlayers(path):
                layer = name + str(filenum)
                filenum += 1
            return path, layer
        return path, name

    def make_geodf(
        yd: Union[gpd.GeoDataFrame, pd.DataFrame],
        igd: Union[gpd.GeoDataFrame, pd.DataFrame],
    ):
        ig = igd.reset_index()
        y = yd.join(ig)
        return GeoDataFrame(y, geometry="geometry")

    #### Main program
    if decimalpoint_german:
        decimal = ","
        separator = ";"
    else:
        decimal = "."
        separator = ","

    # metadata like crs are assigned to the geodataframe
    if dfg is not None and metadata is not None:
        if isinstance(dfg, gpd.GeoDataFrame):
            if metadata is not None:
                dfg.set_crs = metadata
    # if dfg is not None, ydf will be added as a column to this dataframe
    # else ydf will be the output
    if dfg is None or len(ydf.columns) > 1:  # ydf to export:   ydf more then 1 column or not to add to dfg
        if nanmask is None:  #
            # out = gpd.GeoDataFrame(ydf)
            out = ydf
        else:
            # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
            v = 0
            dfnew = deepcopy(ydf[0:0])
            empty = deepcopy(ydf[:1])
            empty.iloc[:] = np.nan
            for cel in nanmask.iloc[:, 0]:
                if cel == True:
                    dfnew = pd.concat([dfnew, pd.DataFrame(empty)], axis=0, ignore_index=True)
                else:
                    dfnew = pd.concat([dfnew, pd.DataFrame(ydf.iloc[v]).T], axis=0, ignore_index=True)
                    v += 1
            out = dfnew
        # if ydf ist the output and geometry-columns exists (in igdf):
        if not (isinstance(out, (gpd.GeoDataFrame))):
            if (igdf is not None) and ("geometry" in igdf.columns):
                out = make_geodf(ydf, igdf)
    else:  # if ydf with one column (result) and to add to dfg
        out = dfg
        # next result field
        nfield = "result"
        fieldnum = 1
        # dfg contains a column with the name of ydf column, the name will be changed
        if nfield in dfg.columns:
            fieldnum = 1
            while nfield in out.columns:
                nfield = nfield + str(fieldnum)
                fieldnum += 1

        if nanmask is None:  #
            out[nfield] = ydf.to_numpy()
        else:  # if nodata-removement was made:
            # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
            v = 0
            lst = []
            for cel in nanmask.iloc[:, 0]:
                if cel == True:
                    lst.append(np.NaN)
                else:
                    lst.append(ydf.iloc[v, 0])  # .values.tolist())
                    v += 1
            out[nfield] = lst
            # append as result-column

    # save dataframe to file or to layer in a geopackage
    if outfile is not None:
        if outextension is None:
            outextension = ""
        elif outextension[0] == ".":
            outextension = outextension[1:]
        if outpath is None:
            outpath = ""
        # if outpath is a geopackage (is.gpkg) a new layer will be stored in this package
        if outpath.endswith(".gpkg"):  # path is geopackage, file ist layer
            path, file = create_layername(outpath, outfile)
            out.columns = out.columns.astype(str)
            if not (isinstance(out, (gpd.GeoDataFrame))):
                raise FileWriteError("Layer is not a Geodataframe.It cans not be written in geopackage ")
            try:
                out.to_file(path, driver="GPKG", layer=file)
            except:
                raise FileWriteError("Layer " + str(file) + " can not be written in geopackage " + str(path))
        else:
            # in the other cases a file (csv, shp or geojson) will be stored to the outpath
            path, file, extension = create_filename(outpath, outfile, outextension, new_version)
            filename = os.path.join(path, file)
            if outextension == "csv":
                try:
                    out.to_csv(filename + extension, header=True, sep=separator, decimal=decimal)
                except:
                    raise FileWriteError("File can not be written: " + str(filename + extension))
            elif outextension == "shp":
                if not (isinstance(out, (gpd.GeoDataFrame))):
                    raise FileWriteError(" Dataframe can not be written as shp because it's not a GEOdataframe")
                try:
                    out.to_file(filename + extension)
                except:
                    raise FileWriteError("File can not be written: " + str(filename + extension))
            elif outextension == "geojson":
                if not (isinstance(out, (gpd.GeoDataFrame))):
                    raise FileWriteError("dfg can not be written as geojson because it's not a GEOdataframe")
                try:
                    out.to_file(filename + extension, driver="GeoJSON")
                except:
                    raise FileWriteError("File can not be written: " + str(filename + extension))
            else:
                raise InvalidParameterValueException("No data output. Wrong extension of the output-file")
    return out


# *******************************
@beartype
def export_featureclass(
    ydf: Union[pd.DataFrame, gpd.GeoDataFrame],  # ydf has to add to XDF
    dfg: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,  # ydf has to be ecxported
    igdf: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    metadata: Optional[Any] = None,
    outpath: Optional[Union[str, pathlib.PosixPath]] = None,
    outfile: Optional[str] = None,  # if not given: pd.DataFrame will given back
    outextension: Optional[str] = None,
    nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,  # german Comma (,) and seperator: ;
    new_version=False,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    """
        Add the result column to the existing geopandas or pandas dataframe (if exists)
        and saved the (geo)dataframe optionaly to a feature class file (like .shp) or to a csv (text) file.
        If the file alredy exists, then the old file will be optionaly deleted or the new file will be named with (file-name)1, 2, ...
        If outpath is a geopackage, then outfile is the name of the new layer.
            If the layer alredy exists, the new layer will be named with (layer-name)1, 2, ...
        If the column name already exists, the new column will be named as result1, result2,... etc.
        If outfile == None: No file will be stored.
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows):
        "True"-cells in nanmask lead to nodata-cells in ydf. So ydf may be added to dfg.
        If no dfg is given:
            no nanmask is needed.
            if ydf has a geometry columns, it may be stored as a GIS-Layer if the extension of the file name is .shp, geojson or gpkg
    Args:
        - ydf: is result of prediction (with one "result"-column)
        - dfg: is the primary feature table (dataframe or geodataframe) - to add with ydf
        - igdf: is contains identfier and geometry column.
                If dgf is empty on this way ydf can become a geodatbase
        - metadata: in case of a geodataframe: crs
        - outpath (optional): Path or geodatapacke of the output-file
        - outfile (optional): Name of file or layer of the output
        - outextension (optional): Name of the file extension (like .shp, geojson or gpkg)
        - nodata (optional): marked rows witch are be droped because of nodata in the prediction input.
        - decimalpoint_german: (optional): default False, german decimal comma (,) and separator (;)
        - new_version: = True: is the output file exists, then a new filename will be generated with the next number.
                       = False:  the existing file will be deleted
    Returns:
        gpd.GeoDataFrame or pd.DataFrame
        optionaly stored as a file (csv, shp or geojson or feature class layer for a geopackage)
    """

    # Argument evaluation
    if dfg is not None and nanmask is None:
        if ydf.shape[0] != dfg.shape[0]:  # rows (.index)
            raise InvalidParameterValueException("ydf and dfg have not the same number of rows")
    if dfg is not None and nanmask is not None:
        if dfg.shape[0] != nanmask.shape[0]:  # rows (.index)
            raise InvalidParameterValueException("dfg and nanamask have not the same number of rows")

    out = _export_featureclass(
        ydf=ydf,
        dfg=dfg,
        igdf=igdf,
        metadata=metadata,
        outpath=outpath,
        outfile=outfile,
        outextension=outextension,
        nanmask=nanmask,
        decimalpoint_german=decimalpoint_german,
        new_version=new_version,
    )
    return out
