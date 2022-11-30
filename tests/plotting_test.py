# -*- coding: utf-8 -*-
"""
Tests for EIS exploratory analyses plotting functions. Unit testing plots is always iffy at best, 
so we'll just stick to checking if the plot return type is correct, and there are no errors.
"""

import numpy as np
from pathlib import Path
from matplotlib import collections as mplc
from matplotlib import pyplot as plt
from eis_toolkit.exploratory_analyses.plotting import histogram
from eis_toolkit.exploratory_analyses.plotting import boxplot
from eis_toolkit.exploratory_analyses.plotting import scatterplot



#multiband.csv: csv data to read in as ndarray. This is output data from raster_to_pandas function.
parent_dir = Path(__file__).parent
csv_path = parent_dir.joinpath("data/remote/multiband.csv")
my_data = np.genfromtxt(csv_path, delimiter=',')



def test_histogram():
    """Test histogram function. Only compares return type."""
    hist= histogram(my_data,1)
    assert type (hist) is tuple
    plt.close()

def test_scatterplot():
    """Test histogram function. Only compares return type."""
    scatter= scatterplot(my_data,1,2)
    assert type (scatter) is mplc.PathCollection
    plt.close()
    
def test_boxplot():
    """Test histogram function, with drawing a single data column. Only compares return type."""
    box= boxplot(my_data,1)
    assert type (box) is dict    
    plt.close()
    
def test_boxplot_multi_column():
    """Test histogram function, with drawing multiple data columns. Only compares return type."""
    my_data_2=my_data[:,1:5]#select only data columns
    box= boxplot(my_data_2)
    assert type (box) is dict    
    plt.close()