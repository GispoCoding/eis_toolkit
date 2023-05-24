from pathlib import Path
import geopandas as gpd

from eis_toolkit.spatial_analyses.cba import CBA

parent_dir = Path(__file__).parent
vector_path = str(parent_dir.joinpath("data/remote/Test_Litho.shp"))
points_path = str(parent_dir.joinpath("data/remote/Test_Occ.shp"))
lines_path = str(parent_dir.joinpath("data/remote/Test_Faults.shp"))
matrix_path = str(parent_dir.joinpath("data/remote/Test_CBA_matrix.shp"))

vector_file = gpd.GeoDataFrame.from_file(vector_path)
points_file = gpd.GeoDataFrame.from_file(points_path)
lines_file = gpd.GeoDataFrame.from_file(lines_path)


def test_crs():
    """Test that Coordinate Systems for input dataset and output grids are equivalent."""
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert cba_grid.cba.crs == shp_cba_grid.crs
    assert cba_grid.cba.crs == vector_file.crs


def test_gridding():
    """Test that cells indexing is coherent."""
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba.index == shp_cba_grid.cba.index).all()


def test_code_envs():
    """Test that binary code produced are coherent."""
    names = list(vector_file.Litho.unique())
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba[names] == shp_cba_grid.cba[names]).all().all()


def test_add_points():
    """Test the add_layer() function for points vector file."""
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    cba_grid.add_layer(
        geodataframe=points_file,
        column="",
        subset_of_target_attribut_values=None,
        Name="Occ",
        buffer=False,
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba["Occ"] == shp_cba_grid.cba["Occ"]).all()


def test_add_points_buffered():
    """Test the add_layer() function for points vector file with buffer option."""
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    cba_grid.add_layer(
        geodataframe=points_path, column="", subset_of_target_attribut_values=None, Name="Occ", buffer=4000
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba["Occ"] == shp_cba_grid.cba["Occ_B"]).all()


def test_add_lines():
    """Test the add_layer() function for mutltilines vector file."""
    cba_grid = CBA()
    cba_grid.init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_of_target_attribut_values="all"
    )
    cba_grid.add_layer(
        geodataframe=lines_file,
        target_attribut="Type",
        subset_of_target_attribut_values=["Thrust", "Normal"],
        Name=None,
        buffer=False,
    )
    shp_cba_grid = CBA()
    shp_cba_grid = shp_cba_grid.from_vector_file(matrix_path)
    assert (cba_grid.cba.Normal == shp_cba_grid.cba.Normal).all()
    assert (cba_grid.cba.Thrust == shp_cba_grid.cba.Thrust).all()
