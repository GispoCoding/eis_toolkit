
from pathlib import Path

import pandas as pd
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.conversions.import_grid import import_grid
# from eis_toolkit.exceptions import FileReadError, InvalidParameterValueException, MatchingRasterGridException

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)

# # input from GUI:
parent_dir = Path(__file__).parent.parent
# name_K = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif"))
# name_Th = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif"))
# name_U = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif"))
# name_target = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif"))
# name_wrong = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_wrong.tif"))
# size_wrong = str(parent_dir.joinpath(r"data/local/data/multiband.tif"))

# # grids and grid-types for X (training based on tif-files)
# grids = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "v"},
#     {"name": "Thorium", "file": name_Th, "type": "v"},
#     {"name": "Uran", "file": name_U, "type": "v"},
# ]

# gridwrong = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "v"},
#     {"name": "Thorium", "file": name_Th, "type": "v"},
#     {"name": "Uran", "file": name_wrong, "type": "v"},
# ]

# gridsize = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "v"},
#     {"name": "Thorium", "file": name_Th, "type": "v"},
#     {"name": "Uran", "file": size_wrong, "type": "v"},
# ]

# gridtype1 = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "t"},
#     {"name": "Thorium", "file": name_Th, "type": "v"},
#     {"name": "Uran", "file": name_wrong, "type": "v"},
# ]

# gridtype2 = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "n"},
#     {"name": "Thorium", "file": name_Th, "type": "n"},
#     {"name": "Uran", "file": name_wrong, "type": "d"},
# ]

name_tif1 = str(parent_dir.joinpath(r"data/test1.tif"))
name_tif2 = str(parent_dir.joinpath(r"data/test2.tif"))
name_tif3 = parent_dir.joinpath(r"data/test1.tif")

grids = [
    {"name": "targe", "type": "t", "file": name_tif1},
    {"name": "test1", "file": name_tif2, "type": "v"},
    {"name": "test2", "file": name_tif3, "type": "v"},
]


# ###############################
def test_import_grid_ok():
    """Test functionality: import of tif files and creating of X as DataFrame."""
    columns, df, metadata = import_grid(grids=grids)

    assert isinstance(columns, dict)
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 0
    assert len(df.index) > 0
    assert len(df.columns) > 0
    assert isinstance(metadata, dict)
    assert metadata["height"] > 0


def test_import_grid_wrong():
    """Test functionality with wrong filenames."""
    with pytest.raises(BeartypeCallHintParamViolation):
        columns, df, metadata = import_grid(grids=6)
    # """Test functionality with wrong filenames."""
    # with pytest.raises(FileReadError):
    #     columns, df, metadata = import_grid(grids=gridwrong)
    # """Test functionality with wrong imagesize/crs."""
    # with pytest.raises(MatchingRasterGridException):
    #     columns, df, metadata = import_grid(grids=gridsize)
    # with pytest.raises(InvalidParameterValueException):
    #     columns, df, metadata = import_grid(grids=gridtype2)
    # with pytest.raises(InvalidParameterValueException):
    #     columns, df, metadata = import_grid(grids=gridtype1)


test_import_grid_ok()
test_import_grid_wrong()
