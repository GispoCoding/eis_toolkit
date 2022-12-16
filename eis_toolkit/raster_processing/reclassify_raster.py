import sys
import rasterio
import numpy as np
import pandas as pd
sys.path.insert(0, "..")


def raster_value_to_int(src: rasterio.io.DatasetReader, band_number: int) -> rasterio.io.DatasetReader:
