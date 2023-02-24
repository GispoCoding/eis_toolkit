import pytest
import rasterio
import pandas as pd
import random
import string
from pathlib import Path


from eis_toolkit.transformations import logarithmic


parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")

def create_example_df():
    with rasterio.open(raster_path) as raster:
        raster = raster.read()
        raster = raster.reshape((4, -1)).T

        string_col = [random.choice(string.ascii_lowercase) for _ in range(raster.shape[0])]
        dataframe = pd.DataFrame(raster, columns=["band_1", "band_2", "band_3", "band_4"])
        dataframe["string_1"] = string_col 
        
    return dataframe


def test_logarithmic_raster():
    """Test that binarize function works as intended."""
    with rasterio.open(raster_path) as raster:
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=None, base=[2], nodata=None, method="replace")

        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[3], base=[2], nodata=None, method="replace")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[1, 3], base=[2], nodata=None, method="replace")

        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[1, 3, 4], base=[2, 10, 2], nodata=None, method="replace")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[1, 2, 3, 4], base=[2, 2, 10, 10], nodata=[-10], method="replace")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[1, 2, 3, 4], base=[2, 2, 10, 10], nodata=[2.749, None, None, None], method="replace")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[3, 1, 4], base=[2, 2, 10], nodata=[1.771, 2.749, -0.766], method="replace")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[3, 1], base=[10], nodata=None, method="extract")
        
        array, meta, settings = logarithmic.log_transform(in_data=raster, selection=[3, 1, 4], base=[2, 2, 10], nodata=[1.771, None, -0.766], method="extract")
        

def test_logarithmic_dataframe():
        """Test that binarize function works as intended."""
        dataframe = create_example_df()
        
        array, columns, settings = logarithmic.log_transform(in_data=dataframe, selection=None, base=[2], nodata=None)
        
        array, columns, settings = logarithmic.log_transform(in_data=dataframe, selection=["band_4", "band_1", "band_2", "band_3"], base=[2, 2, 10, 10], nodata=[None, 2.749, None, None])




