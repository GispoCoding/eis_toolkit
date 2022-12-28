from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.raster_processing.unique_combinations import unique_combinations
from eis_toolkit.exceptions import InvalidParameterValueException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
output_raster_path = parent_dir.joinpath("data/remote/small_raster - Copy.tif")

def test_unique_combinations():
    """Test unique combinations."""
    src = rasterio.open(raster_path)
    src_2 = rasterio.open(raster_path)

    unique_combinations_raster = unique_combinations([src, src_2], output_raster_path)

    assert len(src.read(1)) == len(unique_combinations_raster.read(1))
    assert src.read(1) is not unique_combinations_raster.read(1)

