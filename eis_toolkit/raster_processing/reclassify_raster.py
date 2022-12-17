from typing import List

import rasterio
import numpy as np

def raster_value_to_np_int(src: rasterio.io.DatasetReader, band_numbers: List[int]) -> rasterio.io.DatasetWriter:

    # Read the raster data into a NumPy array
    data = src.read()
    
    # Change the data type to int
    data = data.astype(np.int)
    
    # Update the data type in the dataset object
    src.meta['dtype'] = np.int
    
    src.write_transform(band_numbers)