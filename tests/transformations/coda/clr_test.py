import numpy as np
import pandas as pd

from eis_toolkit.transformations.coda.clr import clr_transform

SINGLE_ROW_DATAFRAME = pd.DataFrame(np.array([1, 1, 1, 2])[None], columns=["a", "b", "c", "d"])

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])

ZEROS_DATAFRAME_4x4 = pd.DataFrame(np.zeros((4, 4)), columns=["V1", "V2", "V3", "V4"])

sample_array = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
SAMPLE_DATAFRAME = pd.DataFrame(sample_array, columns=["a", "b", "c", "d"])


def test_clr_transform_simple():
    """Test CLR transform core functionality."""
    result = clr_transform(ONES_DATAFRAME_4x4)
    pd.testing.assert_frame_equal(result, ZEROS_DATAFRAME_4x4, check_names=True)


# def test_inverse_clr_simple():
#     """TODO: docstring."""
#     result, scale = inverse_clr(ZEROS_DATAFRAME_4x4)
#     pd.testing.assert_frame_equal(_scale(result, scale), ONES_DATAFRAME_4x4)  # TODO: call each row with its scale
