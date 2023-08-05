import os
import pathlib

import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Optional, Union

from eis_toolkit.exceptions import FileWriteError


# *******************************
@beartype
def _export_grid(
    df: pd.DataFrame,
    metadata: dict,  # metadata-Dictionary
    outpath: Optional[Union[str, pathlib.PosixPath]] = None,
    outfile: Optional[str] = None,  # if not given: pd.DataFrame will given back
    nanmask: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    def create_filename(
        outpath: Optional[str] = None,
        outfile: Optional[str] = None,  # if not given: pd.DataFrame will given back
        outextension: Optional[str] = None,  # is default 'tif'
    ) -> str:
        if len(outextension) > 0:
            if outextension[0] != ".":
                outextension = "." + outextension
        filenum = 1
        filename = os.path.join(outpath, outfile)
        if (os.path.exists(os.path.abspath(filename + outextension))) or (os.path.exists(os.path.abspath(filename))):
            while (os.path.exists(os.path.abspath(filename + str(filenum) + outextension))) or (
                os.path.exists(os.path.abspath(filename + str(filenum)))
            ):
                filenum += 1
            return filename + str(filenum) + outextension  # open(filename+str(filenum)+'.'+extension,'w')
        return filename + outextension

    # main
    if nanmask is None:  # reshape with metadata width and hiegt (nonaodata-samples ar removed)
        out = df.to_numpy().reshape(metadata["height"], metadata["width"])
    else:
        # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
        v = 0
        lst = []
        for cel in nanmask.iloc[:, 0]:
            if cel is True:
                lst.append(np.NaN)
            else:
                lst.append(df.iloc[v, 0])  # .values.tolist())
                v += 1

        out = np.array(lst).reshape(metadata["height"], metadata["width"])

    # save dataframe to file
    outextension = "tif"
    if outfile is not None:
        if outextension is None:
            outextension = ""
        if outpath is None:
            outpath = ""
        file = create_filename(outpath, outfile, outextension)

        profile = metadata
        profile.update(
            {"count": 1, "driver": "GTiff", "dtype": "float32"}  # nicht zwingend float32: ggf. int bei classification
        )

        try:
            with rasterio.open(file, "w", **profile) as dst:
                dst.write_band(1, out.astype(rasterio.float32))
        except:  # noqa: E722
            raise FileWriteError("File can not be written: " + str(file))

    return out


# *******************************
@beartype
def export_grid(
    df: pd.DataFrame,
    metadata: dict,
    outpath: Optional[Union[str, pathlib.PosixPath]] = None,
    outfile: Optional[str] = None,  # if not given: pd.DataFrame will given back
    nanmask: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
        Write a dataframe to one or more images.

        Reshape one column of the pandas DataFrame to a new dataframe and store this in a raster file (tif).
        In case a nanmask is availabel (nan-cells for prediction input caused droped rows):
           "True"-cells in nanmask lead to nodata-cells in the output dataframe (y).
        Metadata contains width and height values out of input grids for prediction,
           as well as the crs (coordinate refrnce system).
        Nodata marks rows witch are droped because of nodata in the prediction input.
        In case outfile is not None, the dataframe will be saved to a geoTiff-file

    Args:
        - df: is the input dataframe created by prediction-method
        - metadata: contains with and height values as weel as coordinate reference system
        - outpath (optional): Path of the output-file
        - outfile (optional): Name of file of the output
            outfile may contain the complete filename.
            outpath should be empty in this case
        - nanmask (optional): in case nodata-samples are removed during "nodata-replacement"

    Returns:
        np.ndarray: 2-d-array (numpy) reddy to output as a tiff, grid,...
    """

    out = _export_grid(
        df=df,
        metadata=metadata,
        outpath=outpath,
        outfile=outfile,  # if not given: pd.DataFrame will given back
        nanmask=nanmask,
    )

    return out
