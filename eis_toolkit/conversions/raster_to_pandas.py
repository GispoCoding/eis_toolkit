from typing import List, Optional, Union

import numpy as np
import pandas as pd
import rasterio


def _raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Union[int, List[int]]] = None,
    names: Optional[List[str]] = None,
    add_img_coord: bool = False,
) -> pd.DataFrame:

    if bands is not None:
        data_array = raster.read(bands)
    else:
        data_array = raster.read()

    if names is None:
        if not all(raster.descriptions):
            names = ["band_" + str(i) for i in range(1, raster.count + 1)]
        else:
            names = raster.descriptions

    row, col = np.where(np.full(data_array.shape[1:], True))
    pixel_data = data_array[..., row, col].T

    if add_img_coord is True:
        data_with_coord = np.column_stack((pixel_data, np.column_stack((row, col))))
        data_frame = pd.DataFrame(data_with_coord, columns=names + ["row", "col"])
    else:
        data_frame = pd.DataFrame(pixel_data, columns=names)

    return data_frame


def raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Union[int, List[int]]] = None,
    band_names: Optional[List[str]] = None,
    add_img_coord: bool = False,
) -> pd.DataFrame:
    """Convert raster to pandas DataFrame.

    If bands are not given, all bands are used for conversion. If band names are not given, band names are read from
    raster meta data. If raster meta data doesn't contain names, then bands are named as band_1, band_2,...,band_n.
    If wanted, image coordinates (row, col) for each pixel can be written to dataframe by setting add_img_coord to True.

    Args:
        raster (rasterio.io.DatasetReader):: Raster to be converted.
        bands (Union[int, List[int]], optional): Selected band(s) from multiband raster. Can be single band index or
        list of band indices. Defaults to None.
        band_names (List[str], optional): Band names. Defaults to None.
        add_img_coord (bool): Determines if pixel coordinates are written into dataframe. Defaults to false.

    Returns:
        pd.DataFrame: Raster converted to pandas dataframe
    """
    if bands is not None:
        if not isinstance(bands, int):
            if not all(isinstance(band, int) for band in bands):
                raise InvalidBandValue

    data_frame = _raster_to_pandas(
        raster=raster,
        bands=bands,
        names=band_names,
        add_img_coord=add_img_coord,
    )
    return data_frame


rast = rasterio.open("tests/data/remote/small_raster.tif")
rast_data_array = rast.read(1)

multiband = "tests/data/local/data/multiband.tif"
meta = rast.meta.copy()
meta["count"] = 4
with rasterio.open(multiband, "w", **meta) as dest:
    for band in range(1, 5):
        dest.write(rast_data_array - band, band)
        dest.set_band_description(band, "band_" + str(band))

multi = rasterio.open("tests/data/local/data/multiband.tif")
