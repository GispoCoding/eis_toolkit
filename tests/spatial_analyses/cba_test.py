# -*- coding: utf-8 -*-

"""Created on Thu Jun  1 09:28:42 2023.

@author: vella.
"""

from pathlib import Path

import geopandas as gpd

import eis_toolkit.spatial_analyses.cba as cba

# import os
# import sys
# os.environ['USE_PYGEOS'] = '0'
# sys.path.append(r"C:\Users\vella\Desktop\EIS_Toolkit\eis_toolkit\eis_toolkit\spatial_analyses")
# from cba import CBA

parent_dir = Path(__file__).parent.parent
# parent_dir = Path(r"C:/Users/vella/Desktop/EIS_Toolkit/eis_toolkit/tests/")

vector_path = str(parent_dir.joinpath("data/remote/Test_Litho.geojson"))
points_path = str(parent_dir.joinpath("data/remote/Test_Occ.geojson"))
lines_path = str(parent_dir.joinpath("data/remote/Test_Faults.geojson"))
matrix_path = str(parent_dir.joinpath("data/remote/Test_CBA_matrix.geojson"))

vector_file = gpd.GeoDataFrame.from_file(vector_path)
points_file = gpd.GeoDataFrame.from_file(points_path)
lines_file = gpd.GeoDataFrame.from_file(lines_path)


def test_crs():
    """Test that Coordinate Systems for input dataset and output grids are equivalent."""
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert cba_grid.crs == shp_cba_grid.crs
    assert cba_grid.crs == vector_file.crs


def test_gridding():
    """Test that cells indexing is coherent."""
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert (cba_grid.index == shp_cba_grid.index).all()


def test_code_envs():
    """Test that binary code produced are coherent."""
    names = list(vector_file.Litho.unique())
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert (cba_grid[names] == shp_cba_grid[names]).all().all()


def test_add_points():
    """Test the add_layer() function for points vector file."""
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    cba_grid = cba.add_layer(
        cba_grid,
        grid,
        geodataframe=points_file,
        column="",
        subset_of_target_attribute_values=None,
        name="Occ",
        buffer=False,
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert (cba_grid["Occ"] == shp_cba_grid["Occ"]).all()


def test_add_points_buffered():
    """Test the add_layer() function for points vector file with buffer option."""
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    cba_grid = cba.add_layer(
        cba_grid,
        grid,
        geodataframe=points_file,
        column="",
        subset_of_target_attribute_values=None,
        name="Occ",
        buffer=4000,
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert (cba_grid["Occ"] == shp_cba_grid["Occ_B"]).all()


def test_add_lines():
    """Test the add_layer() function for mutltilines vector file."""
    grid, cba_grid = cba.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribute_values="all"
    )
    cba_grid = cba.add_layer(
        cba_grid,
        grid,
        geodataframe=lines_file,
        column="Type",
        subset_of_target_attribute_values=["Thrust", "Normal"],
        name=None,
        buffer=False,
    )
    shp_cba_grid = cba.from_vector_file(matrix_path)
    assert (cba_grid.Normal == shp_cba_grid.Normal).all()
    assert (cba_grid.Thrust == shp_cba_grid.Thrust).all()
