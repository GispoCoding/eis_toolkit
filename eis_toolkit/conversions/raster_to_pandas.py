from typing import List, Union
import rasterio
import pandas as pd
import numpy as np
from typing import List


def raster_to_pandas(
    raster: rasterio.io.DatasetReader,
    bands: Union[int, List[int]] = None,
    names: List[str] = None,
    add_img_coord: bool = False,
) -> pd.DataFrame:

    if bands is not None:
        arr = raster.read(bands)
    else:
        arr = raster.read()

    # Check for existing band names
    if names is None:
        if not all(rast.descriptions):
            names = ["band_" + str(i) for i in range(1, raster.count + 1)]
        else:
            names = raster.descriptions

    row, col = np.where(np.full(arr.shape[1:], True))  # Image coords for pixels
    pixel_data = arr[..., row, col].T  # Read data from image coords

    if add_img_coord == True:
        data_with_coord = np.column_stack((pixel_data, np.column_stack((row, col))))
        df = pd.DataFrame(data_with_coord, columns=names + ["row", "col"])
    else:
        df = pd.DataFrame(pixel_data, columns=names)

    return df


rast = rasterio.open("tests/data/remote/small_raster.tif")
rast_arr = rast.read(1)

multiband = "tests/data/local/data/multiband.tif"
meta = rast.meta.copy()
meta["count"] = 4
with rasterio.open(multiband, "w", **meta) as dest:
    for band in range(1, 5):
        dest.write(rast_arr - band, band)
        dest.set_band_description(band, "band_" + str(band))

multi = rasterio.open("tests/data/local/data/multiband.tif")
