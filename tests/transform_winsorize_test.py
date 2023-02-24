import pytest
import rasterio
import pandas as pd
import random
import string
from pathlib import Path


from eis_toolkit.transformations import winsorize


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


def test_winsorize_raster():
    """Test that binarize function works as intended."""
    with rasterio.open(raster_path) as raster:
        # Option "absolute"
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=None, limits=[(0, 2)], 
                                                    replace_type="absolute", replace_values=[(0, 1)], nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1], limits=[(0, 2)], 
                                                    replace_type="absolute", replace_values=[(0, 1)], nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3], limits=[(0, 2)], 
                                                    replace_type="absolute", replace_values=[(0, 1)], nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3, 4], limits=[(0, 2), (-1, 2), (1, 3)], 
                                                    replace_type="absolute", replace_values=[(-1, 0)], nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3, 4], limits=[(0, 2)], 
                                                    replace_type="absolute", replace_values=[(-1, 1), (-10, 10), (-100, 100)], nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 2, 3, 4], limits=[(0, 2), (-1, 2), (1, 3), (2, 4)], 
                                                    replace_type="absolute", replace_values=[(-1, 1), (-10, 10), (-100, 100), (-1000, 1000)], 
                                                    nodata=[2.749, None, None, None], method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1, 4], limits=[(1, 3), (0, 2), (2, 4)], 
                                                    replace_type="absolute", replace_values=[(-100, 100), (-1, 1), (-1000, 1000)], 
                                                    nodata=[1.771, 2.749, -0.766], method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1], limits=[(0, 1)], 
                                                    replace_type="absolute", replace_values=[(99, 999)], nodata=None, method="extract")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1, 4], limits=[(1, 3), (0, 2), (2, 4)], 
                                                    replace_type="absolute", replace_values=[(-100, 100), (-1, 1), (-1000, 1000)], 
                                                    nodata=[1.771, None, 0], method="extract")
        
        # Option "percentiles"
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=None, limits=[(10, 10)], 
                                                    replace_type="percentiles", replace_position=["outside"], 
                                                    nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1], limits=[(10, 5)], 
                                                    replace_type="percentiles", replace_position=["outside"], 
                                                    nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3], limits=[(5, 10)], 
                                                    replace_type="percentiles", replace_position=["outside"], 
                                                    nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3, 4], limits=[(10, 10), (10, 5), (5, 10)],
                                                    replace_type="percentiles", replace_position=["outside"], 
                                                    nodata=None, method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 3, 4], limits=[(5, 5)],
                                                    replace_type="percentiles", replace_position=["outside", "outside", "inside"], 
                                                    nodata=None, method="replace")

        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[1, 2, 3, 4], limits=[(10, 10), (10, 5), (5, 10), (25, 25)],
                                                    replace_type="percentiles", replace_position=["outside", "outside", "inside", "inside"], 
                                                    nodata=[2.749, None, None, None], method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1, 4], limits=[(10, 10), (10, 5), (5, 10)],
                                                    replace_type="percentiles", replace_position=["outside", "outside", "inside"], 
                                                    nodata=[1.771, 2.749, -0.766], method="replace")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1], limits=[(10, 10)],
                                                    replace_type="percentiles", replace_position=["outside"], 
                                                    nodata=None, method="extract")
        
        array, meta, settings = winsorize.winsorize(in_data=raster, selection=[3, 1, 4], limits=[(10, 10), (10, 5), (5, 10)],
                                                    replace_type="percentiles", replace_position=["outside", "outside", "inside"], 
                                                    nodata=[1.771, None, 0], method="extract")


def test_winsorize_dataframe():
        """Test that binarize function works as intended."""
        dataframe = create_example_df()
        
        array, columns, settings = winsorize.winsorize(in_data=dataframe, selection=None, limits=[(10, 10)],
                                                        replace_type="percentiles", replace_position=["outside"],
                                                        nodata=None)

        array, columns, settings = winsorize.winsorize(in_data=dataframe, selection=["band_3", "band_1", "band_4"], limits=[(10, 10), (10, 5), (5, 10)],
                                                        replace_type="percentiles", replace_position=["outside", "outside", "inside"],
                                                        nodata=[1.771, None, 0])

