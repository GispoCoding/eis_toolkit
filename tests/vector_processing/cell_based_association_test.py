from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy
import pytest
import rasterio

import eis_toolkit.vector_processing.cell_based_association as cba
from eis_toolkit.exceptions import EmptyDataFrameException, InvalidColumnException, InvalidParameterValueException

with TemporaryDirectory() as tmp_dir:
    output_file = Path(tmp_dir + "_cba_matrix")

parent_dir = Path(__file__).parent.parent

vector_path = str(parent_dir.joinpath("data/remote/Test_Litho.geojson"))
points_path = str(parent_dir.joinpath("data/remote/Test_Occ.geojson"))
lines_path = str(parent_dir.joinpath("data/remote/Test_Faults.geojson"))
matrix_path = str(parent_dir.joinpath("data/remote/Test_CBA_matrix.geojson"))
raster_path = str(parent_dir.joinpath("data/remote/Test_CBA_matrix_check.tif"))

vector_file = gpd.GeoDataFrame.from_file(vector_path)
points_file = gpd.GeoDataFrame.from_file(points_path)
lines_file = gpd.GeoDataFrame.from_file(lines_path)


def test_crs():
    """Test that Coordinate Systems for input dataset and output grids are equivalent."""
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert cba_grid.crs == shp_cba_grid.crs
    assert cba_grid.crs == vector_file.crs


def test_gridding():
    """Test that cells indexing is coherent."""
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert (cba_grid.index == shp_cba_grid.index).all()


def test_code_envs():
    """Test that binary code produced are coherent."""
    names = list(vector_file.Litho.unique())
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert (cba_grid[names] == shp_cba_grid[names]).all().all()


def test_add_points():
    """Test the add_layer() function for points vector file."""
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    cba_grid = cba._add_layer(
        cba_grid,
        grid,
        geodataframe=points_file,
        column="",
        subset_target_attribute_values=None,
        name="Occ",
        buffer=False,
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert (cba_grid["Occ"] == shp_cba_grid["Occ"]).all()


def test_add_points_buffered():
    """Test the add_layer() function for points vector file with buffer option."""
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    cba_grid = cba._add_layer(
        cba_grid,
        grid,
        geodataframe=points_file,
        column="",
        subset_target_attribute_values=None,
        name="Occ",
        buffer=4000,
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert (cba_grid["Occ"] == shp_cba_grid["Occ_B"]).all()


def test_add_lines():
    """Test the add_layer() function for mutltilines vector file."""
    grid, cba_grid = cba._init_from_vector_data(
        cell_size=4000, geodataframe=vector_file, column="Litho", subset_target_attribute_values=None
    )
    cba_grid = cba._add_layer(
        cba_grid,
        grid,
        geodataframe=lines_file,
        column="Type",
        subset_target_attribute_values=["Thrust", "Normal"],
        name=None,
        buffer=False,
    )
    shp_cba_grid = cba._from_vector_file(matrix_path)
    assert (cba_grid.Normal == shp_cba_grid.Normal).all()
    assert (cba_grid.Thrust == shp_cba_grid.Thrust).all()


def test_cba():
    """Test the cell_based_association() function to produce raster grid."""
    cba.cell_based_association(
        cell_size=5000,
        geodata=[vector_file, points_file, points_file, lines_file, lines_file, lines_file],
        output_path=str(output_file),
        column=["Litho", "", "", "Type", "Type", "Test"],
        subset_target_attribute_values=[None, None, None, [], ["Thrust"], [1, 2]],
        add_name=["Occ", "Occ_B", "All_", "F_", ""],
        add_buffer=[False, 5000, 5000, False, 5000],
    )
    with rasterio.open(raster_path, "r") as one:
        with rasterio.open(str(output_file) + ".tif", "r") as two:
            one_array = one.read()
            two_array = two.read()
    numpy.testing.assert_equal(one_array, two_array)


def test_empty_geodata():
    """Test that empty Geodataframe raises the correct exception."""
    empty_gdf = gpd.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        cba.cell_based_association(cell_size=5000, geodata=[empty_gdf], output_path=str(output_file), column=["Litho"])


def test_invalid_cell_size():
    """Test that null and negative cell size raise the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        cba.cell_based_association(
            cell_size=-5000, geodata=[vector_file], output_path=str(output_file), column=["Litho"]
        )
    with pytest.raises(InvalidParameterValueException):
        cba.cell_based_association(cell_size=0, geodata=[vector_file], output_path=str(output_file), column=["Litho"])


def test_invalid_buffer():
    """Test that negative buffer size raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        cba.cell_based_association(
            cell_size=5000,
            geodata=[vector_file, points_file],
            output_path=str(output_file),
            column=["Litho", ""],
            add_buffer=[-5000],
        )


def test_invalid_column():
    """Test that invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        cba.cell_based_association(
            cell_size=5000, geodata=[vector_file], output_path=str(output_file), column=["RandomName"]
        )


def test_invalid_subset():
    """Test that invalid subset of target attributes raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        cba.cell_based_association(
            cell_size=5000,
            geodata=[vector_file, lines_file],
            output_path=str(output_file),
            column=["Litho", "Type"],
            subset_target_attribute_values=["lorem1", "lorem2"],
        )
