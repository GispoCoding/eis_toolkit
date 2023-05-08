from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Iterable


def _raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Iterable[int]],
    add_coordinates: bool,
) -> pd.DataFrame:

    if bands is not None:
        data_array = raster.read(bands)
        band_names = ["band_" + str(i) for i in bands]
    else:
        data_array = raster.read()
        band_names = ["band_" + str(i) for i in range(1, raster.count + 1)]

    row, col = np.where(np.full(data_array.shape[1:], True))
    pixel_data = data_array[..., row, col].T

    if add_coordinates:
        pixel_data = np.column_stack((pixel_data, np.column_stack((row, col))))
        band_names += ["row", "col"]

    return pd.DataFrame(pixel_data, columns=band_names)


@beartype
def raster_to_pandas(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Iterable[int]] = None,
    add_coordinates: bool = False,
) -> pd.DataFrame:
    """Convert raster to pandas DataFrame.

    If bands are not given, all bands are used for conversion. Selected bands are named based on their index e.g.,
    band_1, band_2,...,band_n. If wanted, image coordinates (row, col) for each pixel can be written to
    dataframe by setting add_coordinates to True.

    Args:
        raster: Raster to be converted.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.
        add_coordinates: Determines if pixel coordinates are written into dataframe. Defaults to False.

    Returns:
        Raster converted to pandas dataframe
    """

    data_frame = _raster_to_pandas(
        raster=raster,
        bands=bands,
        add_coordinates=add_coordinates,
    )
    return data_frame
