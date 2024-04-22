import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import InvalidRasterBandException
from eis_toolkit.utilities.checks.raster import check_raster_bands


def _raster_to_dataframe(
    raster: rasterio.io.DatasetReader, bands: Sequence[int], add_coordinates: bool, nodata_value: Optional[float]
) -> pd.DataFrame:

    if bands is not None:
        data_array = raster.read(bands)
        band_names = ["band_" + str(i) for i in bands]
    else:
        data_array = raster.read()
        band_names = ["band_" + str(i) for i in range(1, raster.count + 1)]

    if nodata_value is None:
        nodata_value = raster.nodata

    if nodata_value is not None:
        valid_data_mask = data_array != nodata_value
    else:
        valid_data_mask = np.full(data_array.shape, True)

    row, col = np.where(np.any(valid_data_mask, axis=0))
    pixel_data = data_array[:, row, col].T

    if add_coordinates:
        pixel_data = np.column_stack((pixel_data, np.column_stack((raster.xy(row, col)))))
        band_names += ["x", "y"]

    return pd.DataFrame(pixel_data, columns=band_names)


@beartype
def raster_to_dataframe(
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    add_coordinates: bool = False,
    nodata_value: Optional[float] = None,
) -> pd.DataFrame:
    """Convert raster to Pandas DataFrame.

    If bands are not given, all bands are used for conversion. Selected bands are named based on their index e.g.,
    band_1, band_2,...,band_n. If wanted, image coordinates (x, y) for each pixel can be written to
    dataframe by setting add_coordinates to True.

    Args:
        raster: Raster to be converted.
        bands: Selected bands from multiband raster. Indexing begins from one. Defaults to None.
        add_coordinates: Determines if pixel coordinates are written into dataframe. Defaults to False.
        nodata_value: Specifies the value to be considered as NoData. If None, raster's nodata value is used.

    Returns:
        Raster converted to a DataFrame.

    Raises:
        InvalidRasterBandException: All selected bands are not contained in the input raster.
    """
    if bands is None or len(bands) == 0:
        bands = range(1, raster.count + 1)
    else:
        if not check_raster_bands(raster, bands):
            raise InvalidRasterBandException(f"Input raster does not contain all selected bands: {bands}.")

    data_frame = _raster_to_dataframe(
        raster=raster,
        bands=bands,
        add_coordinates=add_coordinates,
        nodata_value=nodata_value,
    )
    return data_frame
