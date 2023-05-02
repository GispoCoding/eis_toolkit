from pathlib import Path
from eis_toolkit.spatial_analyses.cba import CBA

import geopandas as gpd
import pytest
import warnings

parent_dir = Path(__file__).parent
vector_path = str(parent_dir.joinpath("data/remote/Test_Litho.shp"))
points_path = str(parent_dir.joinpath("data/remote/Test_Occ.shp"))
lines_path = str(parent_dir.joinpath("data/remote/Test_Faults.shp"))
matrix_path = str(parent_dir.joinpath("data/remote/Test_CBA_matrix.shp"))

def test_crs():
    """Test that Coordinate Systems for input dataset and output grids are equivalent"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    vector_file = gpd.GeoDataFrame.from_file(vector_path)
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert cba_grid.cba.crs == shp_cba_grid.crs
    assert cba_grid.cba.crs == vector_file.crs

def test_gridding():
    """Test that cells indexing is coherent"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba.index == shp_cba_grid.cba.index).all() == True

def test_code_envs():
    """Test that binary code produced are coherent"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    vector_file = gpd.GeoDataFrame.from_file(vector_path)
    names = list(vector_file.Litho.unique())
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba[names] == shp_cba_grid.cba[names]).all().all() == True

def test_add_points():
    """Test the add_layer() function for points vector file"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    cba_grid.add_layer(vector_file_path = points_path,
                   target_attribut='',
                   subset_of_target_attribut_values=None,
                   Name = "Occ",
                   buffer = False)
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba["Occ"] == shp_cba_grid.cba["Occ"]).all() == True

def test_add_points_buffered():
    """Test the add_layer() function for points vector file with buffer option"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    cba_grid.add_layer(vector_file_path = points_path,
                   target_attribut='',
                   subset_of_target_attribut_values=None,
                   Name = "Occ",
                   buffer = 4000)
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba["Occ"] == shp_cba_grid.cba["Occ_B"]).all() == True

def test_add_lines():
    """Test the add_layer() function for mutltilines vector file"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cba_grid = CBA()
    cba_grid.init_from_vector_file(cell_size = 4000,
	                               vector_file_path = vector_path,
	                               target_attribut='Litho',
	                               subset_of_target_attribut_values="all")
    cba_grid.add_layer(vector_file_path = lines_path,
                   target_attribut='Type',
                   subset_of_target_attribut_values=["Thrust","Normal"],
                   Name = None,
                   buffer = False)
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba.Normal == shp_cba_grid.cba.Normal).all() == True
    assert (cba_grid.cba.Thrust == shp_cba_grid.cba.Thrust).all() == True
