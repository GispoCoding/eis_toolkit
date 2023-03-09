
from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import OneHotEncoder
from eis_toolkit.exceptions import InvalidParameterValueException, FileReadWriteError

# *******************************
def _export_grid(
    df: pd.DataFrame,
    metadata: dict,            # metadata-Dictionary
    outpath: Optional [str] = None,  
    outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
    outextension: Optional [str] = 'tif',
    nanmask: Optional[pd.DataFrame] = None
    
) -> np.ndarray:

    def create_filename(
        outpath: Optional [str] = None,  
        outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
        outextension: Optional [str] = None,       # is default 'tif'
    ) -> str:
        if len(outextension) > 0:
            if outextension[0] != '.':
                outextension = '.'+outextension
        filenum = 1
        filename = os.path.join(outpath, outfile)
        if (os.path.exists(os.path.abspath(filename+outextension))) or (os.path.exists(os.path.abspath(filename))):
            while (os.path.exists(os.path.abspath(filename+str(filenum)+outextension))) or (os.path.exists(os.path.abspath(filename+str(filenum)))):
                filenum+=1
            return filename+str(filenum)+outextension   #open(filename+str(filenum)+'.'+extension,'w')
        return filename+outextension

    # main
    if nanmask is None:   # reshape with metadata width and hiegt (nonaodata-samples ar removed)
        # width = metadata.width
        # height = metadata.height
        out = df.to_numpy().reshape(metadata['height'], metadata['width'])
    else:
        # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
        v = 0
        lst = []
        for cel in nanmask.iloc[:,0]:
            if cel == True:
                lst.append(np.NaN)
            else:
                lst.append(df.iloc[v,0])     # .values.tolist())
                v += 1

        out = np.array(lst).reshape(metadata['height'], metadata['width'])

    # save dataframe to file 
    if outfile is not None:
        if outextension is None:
            outextension = ''
        if outpath is None:
            outpath = '' 
        file = create_filename(outpath, outfile, outextension)

        profile = metadata
        profile.update(
             {'count': 1,  
             'driver': 'GTiff',
             'dtype': 'float32'}                  # nicht zwingend float32: ggf. int bei classification
        )                 #dtype=rasterio.uint16,
         #   'tiled': False,
        #     #compress='lzw'
        # )
        try:
            with rasterio.open(file, 'w',**profile) as dst:         #os.path.join(outpath, 'test1k.tif'), 'w', **profile) as dst:
                dst.write_band(1, out.astype(rasterio.float32))
        except:
            raise FileReadWriteError('Problems with ' + str(dst))
    
    return out

# *******************************
def export_grid(
    df: pd.DataFrame,
    metadata: dict,
    outpath: Optional [str] = None,  
    outfile: Optional [str] = None,            # if not given: pd.DataFrame will given back
    outextension: Optional [str] = None,
    nanmask: Optional[pd.DataFrame] = 'tif'
) -> np.ndarray:

    """ 
        Reshape one column of the pandas df to a new dataframe with width and height.
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
            "True"-cells in nanmask lead to nodata-rows in the output dataframe (y).
        Metadata contains width and height values out of input grids for prediction, as well as the crs (coordinate reference system).
        In case outfile is not None, the dataframe will be saved to a geoTiff-file 
    Args:
        - df (pandas DataFrame): Is the input df, result from prediction-method.
        - metadata (dictionary): Contains with and height values.
        - outpath (string, optional): Path of the output-file
        - outfile (string, optional): Name of file of the output
        - outextension (string, optional): Name of the file extension (like .tif)       
        - nanmask (pandas DataFrame): In case nodata-samples are removed during "nodata-replacement"
    Returns:
        np.array: 2-d-array (numpy) 
        optionaly: output as a geotiff file
    """

    # Argument evaluation
    fl = []
    if not isinstance(metadata, dict):
        fl.append('Argument metadata is not a Dictionary')
    if not isinstance(df, pd.DataFrame):
        fl.append('Argument df is not a DataFrame') 
    if not ((isinstance(outpath, str) or (outpath is None)) and 
        (isinstance(outfile, str) or (outfile is None)) and 
        (isinstance(outextension, str) or (outextension is None))):
        #raise InvalidParameterValueException ('***  outpath, outfile or outextension is not str (or None)')
        fl.append('Arguments outpath, outfile or outextension are not str or are not None')
    if not ((isinstance(nanmask, pd.DataFrame)) or (nanmask is None)):
        fl.append('Argument nanmask is not a DataFrame and is not None')
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])

    out = _export_grid(
    df = df,
    metadata = metadata,
    outpath = outpath,  
    outfile = outfile,            # if not given: pd.DataFrame will given back
    outextension = outextension,
    nanmask  = nanmask
    )

    return out

