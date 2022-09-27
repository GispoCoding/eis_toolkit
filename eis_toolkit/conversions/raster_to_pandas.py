from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException


def _raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    add_img_coord: bool = False,
) -> pd.DataFrame:

    if bands is not None:
        data_array = raster.read(bands)
        band_names = ["band_" + str(i) for i in bands]
    else:
        data_array = raster.read()
        band_names = ["band_" + str(i) for i in range(1, raster.count + 1)]

    row, col = np.where(np.full(data_array.shape[1:], True))
    pixel_data = data_array[..., row, col].T

    if add_img_coord is True:
        data_with_coord = np.column_stack((pixel_data, np.column_stack((row, col))))
        data_frame = pd.DataFrame(data_with_coord, columns=band_names + ["row", "col"])
    else:
        data_frame = pd.DataFrame(pixel_data, columns=band_names)

    return data_frame


def raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    add_img_coord: bool = False,
) -> pd.DataFrame:
    """Convert raster to pandas DataFrame.

    If bands are not given, all bands are used for conversion. Selected bands are named based on their index e.g.,
    band_1, band_2,...,band_n. If wanted, image coordinates (row, col) for each pixel can be written to
    dataframe by setting add_img_coord to True.

    Args:
        raster (rasterio.io.DatasetReader):: Raster to be converted.
        bands (List[int], optional): Selected bands from multiband raster. Indexing begins from one. Defaults to None.
        add_img_coord (bool): Determines if pixel coordinates are written into dataframe. Defaults to false.

    Returns:
        pd.DataFrame: Raster converted to pandas dataframe
    """
    if bands is not None:
        if not isinstance(bands, list):
            raise InvalidParameterValueException
        elif not all(isinstance(band, int) for band in bands):
            raise InvalidParameterValueException

    data_frame = _raster_to_pandas(
        raster=raster,
        bands=bands,
        add_img_coord=add_img_coord,
    )
    return data_frame
