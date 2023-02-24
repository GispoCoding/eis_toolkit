import pytest
import rasterio
import pandas as pd
import random
import string
from pathlib import Path


from eis_toolkit.transformations import linear


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


def test_linear_z_score_raster():
    """Test that binarize function works as intended."""
    with rasterio.open(raster_path) as raster:
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=None, with_mean=[True], with_sd=[True], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1], with_mean=[True], with_sd=[True], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 3], with_mean=[True], with_sd=[True], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 3, 4], with_mean=[True, True, False], with_sd=[True, True, False], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 3, 4], with_mean=[True, True, False], with_sd=[True], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 3, 4], with_mean=[True], with_sd=[True, True, False], nodata=None, method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 2, 3, 4], with_mean=[True, False, True, False], with_sd=[False, True, False, True], nodata=[-10], method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[1, 2, 3, 4], with_mean=[True, True, False, False], with_sd=[True, True, False, False], nodata=[2.749, None, None, None], method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[3, 1, 4], with_mean=[True, True, False], with_sd=[True, True, False], nodata=[1.771, 2.749, -0.766], method="replace")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[3, 1], with_mean=[True], with_sd=[True], nodata=None, method="extract")
        array, meta, settings = linear.z_score_norm(in_data=raster, selection=[3, 1, 4], with_mean=[True, True, False], with_sd=[True, True, False], nodata=[1.771, None, -0.766], method="extract")
        
        
def test_linear_z_score_dataframe():
        """Test that binarize function works as intended."""
        dataframe = create_example_df()
        
        array, columns, settings = linear.z_score_norm(in_data=dataframe, selection=None, with_mean=[True], with_sd=[True], nodata=None)
        array, columns, settings = linear.z_score_norm(in_data=dataframe, selection=["band_3", "band_1", "band_4"], with_mean=[True, True, False], with_sd=[True, True, False], nodata=[1.771, None, -0.766])

        
        




