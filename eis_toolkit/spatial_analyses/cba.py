# -*- coding: utf-8 -*-

"""Created on Thu Jun  1 08:57:13 2023.

@author: vella.
"""

from numbers import Number

# os.environ['USE_PYGEOS'] = '0'
from typing import Optional, Tuple, Union

# import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import wkt
from shapely.geometry import Polygon


def init_from_vector_data(
    cell_size: int,
    geodataframe: gpd.GeoDataFrame,
    column: str = "",
    subset_of_target_attribute_values: Optional[list] = None,
) -> None:
    """Creation of CBA object.

    Initializes a CBA matrix from a vector file. The mesh is calculated
    according to the geometries contained in this file and the size
    of cells.

    Args:
        cell_size: size of the cells (ex : 3000).
        geodataframe: path to vector file.
        target_attribut: <optionnel> name of the attribute of interest.
            If no attribute is specified, then an artificial attribute is created
            representing the presence or absence of the geometries of this file for
            each cell of the CBA grid. If a target attribute is indicated, it must be
            categorical. Other types of attributes (numerical/string) are not managed.
            A categorical attribute will generate as many columns (binary) in the
            CBA matrix than values considered of interest
            (dummification). See parameter <subset_of_target_attribut_values>
        subset_of_target_attribut_values: <optionnel> list of values
            of interest of the target attribute, in case a categorical target attribute
            has been specified. Allows to filter a subset of relevant values.

    Returns:
        CBA object is initialized.
    """

    # Allows a non-mutable default value
    if subset_of_target_attribute_values is None:
        subset_of_target_attribute_values = []

    # Reading the vector file
    geodata = geodataframe.copy(deep=True)

    # Initialization of the grid
    grid = get_grid(geodata, cell_size, cell_size)
    grid.crs = geodata.crs

    # Filling the grid with overlapped geometries
    indicators, join_grid = prepare_grid(geodataframe, grid, column, subset_of_target_attribute_values)
    tmp = join_grid[["geom"]].copy(deep=True)
    tmp.rename({"geom": "geometry"}, axis=1, inplace=True)

    # Saving results in attributes
    cba = gpd.GeoDataFrame(pd.concat([tmp.groupby(tmp.index).first(), indicators], axis=1))
    cba = cba.set_crs(grid.crs, allow_override=True)
    cba = cba.astype("int", errors="ignore")

    return grid, cba


def add_layer(
    cba: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    geodataframe: gpd.GeoDataFrame,
    column: str = "",
    subset_of_target_attribute_values: Optional[list] = None,
    name: Optional[str] = None,
    buffer: Union[Number, bool] = False,
) -> None:
    """Add new data to CBA matrix.

    Allow to add data to an existing matrix. Mesh size and placement are not
    altered. On or multiple columns are added to the CBA matrix based on
    targeted shapes and/or attributes.

    Args:
        geodataframe: path to vector file.
        target_attribut: <optional> Name of the targeted attribute. If
            no attribute are specified, added data fo each cell are based on the
            absence/presence intersection between cells and the shapes within the
            selected vector file. If specified, attribute must be categorical (list)
            and will generate as much columns as it has unique values in
            <subset_of_target_attribut_values> (dummification). Note: if a file
            contain multiple target attribute, <add_layer> has to be executed as
            many times as there are target attributes in this file.
        Name: <optional> Name of the coluumn to add to the matrix.
        buffer: <optional> Allow the use of a buffer around shapes before
            the intersection with CBA cells.Minimize border effects or allow increasing
            positive samples (i.e. cells with mineralization). Default: False;
            Value (CRS units): size of the buffer.
        subset_of_target_attribut_values: <optionnel> list of categorical
            values of interest within the targeted attributes. Allow filtering subset
            of relevant values.

    Returns:
        CBA object is completed.
    """

    # Allows a non-mutable default value
    if subset_of_target_attribute_values is None:
        subset_of_target_attribute_values = []

    # Prerequisite: the CBA grid must already have been initialized
    assert cba is not None

    # Reading the vector file
    geodata = geodataframe.copy(deep=True)

    # No buffer
    if buffer is False:
        # Recovery of the grid calculated at the initialization of the CBA
        grid = gpd.GeoDataFrame(cba)

        # Adding a column to the CBA
        dummies, join_grid = prepare_grid(geodata, grid, column, subset_of_target_attribute_values)

    # Buffer of specified value
    else:
        # Calculation of the buffer disks
        added_buf = get_disc(cba, grid, buffer)

        # Adding a column to the CBA
        dummies, join_disc = prepare_disc(geodata, added_buf, column, subset_of_target_attribute_values)

    # Application of the indicated name (otherwise filename is used)
    if name is not None:
        dummies.name = name

    # Completion of the CBA object (values and names)
    cba = cba.join(dummies).replace(np.nan, 0)
    if column == "":
        cba[dummies.name] = cba[dummies.name].map(int)

    cba = cba.astype("int", errors="ignore")

    return cba


def prepare_grid(  # type: ignore[no-any-unimported]
    geodata: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    target_attribut: str,
    subset_of_target_attribut_values: list,
) -> Union[list, gpd.GeoDataFrame]:
    """Intermediate utility.

    Preparation of the dummification of selected variables and intersection
    between grid and input dataset.

    Args:
        geodata:  the geodata object corresponding to the vector file.
        grid: initialized mesh.
        target_attribut: target attribute to extract from the vector file.
        subset_of_target_attribut_values: subset of the relevant values
        of the target attribute

    Returns: pair of variables
        -> boolean indicating if the target attribute must be dummified
        -> grid resulting from the intersection between the mesh and the
        vector file
    """

    # Data verification
    dummification, target, identified_values = check_and_prepare_param(
        geodata, target_attribut, subset_of_target_attribut_values
    )

    # Spatial join of the intersection between the grid and the current map
    join_grid = gpd.sjoin(grid, geodata, how="inner", predicate="intersects")
    if dummification:
        tmp = pd.get_dummies(join_grid[join_grid[target].isin(identified_values)][[target]], prefix="", prefix_sep="")
        indicators = tmp.groupby(tmp.index).max()
    else:
        tmp = join_grid[target]
        indicators = tmp.groupby(tmp.index).max()
    return indicators, join_grid


def check_and_prepare_param(  # type: ignore[no-any-unimported]
    geodata: gpd.GeoDataFrame,
    target: str,
    attribut_values: list,
    target_name: Optional[str] = "Added",
) -> Tuple[bool, str, list]:
    """Intermediate utility.

    Prepare the parameters and check their consistency.

    Args:
        filepath: path of the vector file to integrate
        geodata: the geodata object corresponding to the vector file
        target: name of the attribute of interest. If no attribute is
            specified, a synthetic one is produced to represent the absence/presence
            of geometry from this file within each cell of the CBA. If an attribute
            is specified, it must be categorical. In this case, it will produce as
            much columns (binary) in the CBA matrix than listed values. See parameter
            <subset_of_target_attribut_values>. Note: if a vector file contains several
            target attributes, it is necessary to run as many times the <add_layer>
            function than target attributes of this file
        attribut_values: list of values of interest for the target attribute,
            in the case where a specific categorical target has been specified. It
            allows filtering a subset of relevant values. "all": all unique values
            for the target will be used.

    Returns: triplet
        -> dummification: boolean indicating the dummification (processing
            of categorical attribute)
        -> target: targeted variable, can be artificial when only geometries
            are considered.
        -> identified_values: subset of values of interest, possibly artificial
            when only geometries are considered.
    """

    dummification = False
    if target == "":
        # Case where we use geometries (e.g. a point file for mineralization)
        # we add a column filled with 1 (presence).
        target = target_name
        attribut_values = [1]
        geodata[target] = int(1)
    elif not attribut_values:
        # Case where we consider the attribute as it is without dummification
        # (e.g. numeric or string target attribute).
        raise ValueError(
            "target_attribut != '' and subset_of_target_attribut_values == [] (numerical/string) "
            "FORBIDDEN AT THIS TIME as not managed !"
        )
    else:
        # Case of a symbolic attribute with identified subset of values of interest.
        dummification = True
        if attribut_values == "all":
            attribut_values = geodata[target].unique()
    assert target in geodata.columns
    identified_values = pd.Series(attribut_values)
    assert identified_values.isin(geodata[target]).all()
    # FORBIDDEN AT THIS TIME as not managed !!!
    # assert target == "" or len(attribut_values) != 0
    assert len(attribut_values) != 0

    return dummification, target, identified_values


def get_disc(  # type: ignore[no-any-unimported]
    geodata: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, buff_radius: Union[int, float]
) -> gpd.GeoDataFrame:
    """Intermediate utility.

    Defines disks with radii defined from the centroids of the cells of a grid.

    Args:
        geodata: geodata to intersect with the grid.
        grid: grid from which the disks will be created.
        buff_radius: radius for created buffer disks

    Returns:
        GeoDataFrame containing the records from the grid buffers/geodata
        intersection.
    """

    # Deletion of cells that do not intersect shapefile geometries
    cells_w_bound = gpd.sjoin(grid, geodata, how="inner", predicate="intersects")
    cells_w_bound["index"] = cells_w_bound.index
    cells_w_bound = cells_w_bound.dissolve(by="index")
    drop_list = geodata.columns
    drop_list = drop_list.drop("geometry")
    cells_w_bound = cells_w_bound.drop(drop_list, axis="columns")

    # Creation of the centroids of the mesh
    centX = cells_w_bound.centroid.x
    centY = cells_w_bound.centroid.y
    df = pd.DataFrame({"CentX": centX, "CentY": centY})
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.CentX, df.CentY), crs=geodata.crs)

    # Transformation into a disc with a given radius
    disc = gpd.GeoDataFrame.copy(points)
    disc["geom"] = disc.geometry.buffer(buff_radius, resolution=16)

    return gpd.GeoDataFrame(disc, geometry="geom", crs=geodata.crs)


def prepare_disc(  # type: ignore[no-any-unimported]
    geodata: gpd.GeoDataFrame,
    disc: gpd.GeoDataFrame,
    target_attribut: str,
    subset_of_target_attribut_values: list,
) -> Tuple[list, gpd.GeoDataFrame]:
    """Intermediate utility.

    Compute spatial joint and generate the table of presence of the different
    attributes.

    Args:
        geodata: the geodata object corresponding to the vector file.
        disc: existing disk 'paving'.
        target_attribut: target attribute to extract from the vector file.
        subset_of_target_attribut_values: subset of the relevant values
            of the target attribute

    Returns: pair of variables
        -> boolean indicating if the target attribute must be dummified
        -> grid resulting from the intersection between the mesh and the
            vector file
    """
    dummification, target, identified_values = check_and_prepare_param(
        geodata, target_attribut, subset_of_target_attribut_values
    )

    # Spatial join of the intersection between the grid and the current map
    join_grid = gpd.sjoin(disc, geodata, how="inner", predicate="intersects")

    if dummification:
        tmp = pd.get_dummies(join_grid[join_grid[target].isin(identified_values)][[target]], prefix="", prefix_sep="")
        indicators = tmp.groupby(tmp.index).max()
    else:
        tmp = join_grid[target]
        indicators = tmp.groupby(tmp.index).max()
    return indicators, join_grid


def get_grid(  # type: ignore[no-any-unimported]
    working_map: gpd.GeoDataFrame, cell_width: int, cell_height: int
) -> gpd.GeoDataFrame:
    """Intermediate utility.

    Produces a regular grid based on geodata surface.
    Paving begin from the upper left corner (North-West) and go to the South
    and East (up to down, left to right paving).

    Args:
        working_map: geodata used as a 'bounding box'.
        cell_width: width of cells.
        cell_height: height of cells.

    Returns:
        GeoDataFrame containing all the cells paving the geodata area.
    """
    xmin, ymin, xmax, ymax = working_map.total_bounds
    rows = int(np.ceil((ymax - ymin) / cell_height))
    cols = int(np.ceil((xmax - xmin) / cell_width))
    x_left_origin = xmin
    x_right_origin = xmin + cell_width
    y_top_origin = ymax
    y_bottom_origin = ymax - cell_height
    print("Columns: %s" % cols)
    print("Rows: %s" % rows)
    print("Cells: %s" % (cols * rows))
    data = []
    for i in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for j in range(rows):
            polygon = Polygon(
                [
                    (x_left_origin, y_top),
                    (x_right_origin, y_top),
                    (x_right_origin, y_bottom),
                    (x_left_origin, y_bottom),
                ]
            )
            data += [[i + j * cols, i, j, polygon]]
            y_top = y_top - cell_height
            y_bottom = y_bottom - cell_height
        x_left_origin = x_left_origin + cell_width
        x_right_origin = x_right_origin + cell_width
    result = pd.DataFrame(data, columns=["cell_id", "x", "y", "geom"])
    result.set_index("cell_id", inplace=True)

    return gpd.GeoDataFrame(result, geometry="geom", crs=working_map.crs)


def from_csv(input_csv_file_path: str) -> gpd.GeoDataFrame:  # type: ignore[no-any-unimported]
    """Intermediate utility.

    Read a CBA grid from CSV file. CSV file must contain "geometry"
    column in 'Well Known Text' (WKT). If a sequence like "__epsg_XXXXX"
    is contained in the file name, it is used to indicate the coordinate
    system (CRS).

    Args:
        input_csv_file_path: CSV filename.

    Returns:
        Corresponding CBA object.
    """

    tmp = input_csv_file_path.split("__")
    crs = tmp[-1].split(".")[0].replace("_", ":")
    df = pd.read_csv(input_csv_file_path)
    df.insert(0, "geometry", df["geom"].apply(wkt.loads))
    # df["geometry"] = df["geom"].apply(wkt.loads)
    df = df.drop("geom", axis=1)
    cba_grid = gpd.GeoDataFrame(df, geometry="geometry")
    cba_grid = cba_grid.set_crs(crs)
    cba_grid.index = df["cell_id"]
    cba_grid = cba_grid.drop("cell_id", axis=1)
    return cba_grid


def from_vector_file(cba_vector_file_path: str) -> gpd.GeoDataFrame:  # type: ignore[no-any-unimported]
    """Intermediate utility.

    Read a CBA grid from a vector file (ESRI shapefile, geojson...).
    All attributes (and all values of each attribute) are kept.
    The geometries contained in the file are assumed to represent a regular
    square grid.

    Args:
        cba_vector_file_path: input vector file.

    Returns:
        Corresponding CBA object.
    """
    geodata = gpd.GeoDataFrame.from_file(cba_vector_file_path)
    cba_grid = gpd.GeoDataFrame(geodata, geometry="geometry")
    cba_grid.set_index("cell_id", inplace=True)
    cba_grid.crs = geodata.crs
    cols = cba_grid.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cba_grid = cba_grid[cols]
    return cba_grid


def to_csv(cba: gpd.GeoDataFrame, output_path: str) -> None:
    """Intermediate utility.

    Saves the object in a CSV file.

    Args:
        output_csv_file_path: name of the saved file suffixed with the
            CRS (spatial projection system).

    Returns:
        None
    """

    crs_txt = f"epsg:{cba.crs.to_epsg()}"
    tmp = gpd.GeoDataFrame(cba)
    tmp["geom"] = tmp.geometry.apply(lambda x: wkt.dumps(x))
    tmp = tmp.drop("geometry", axis=1)
    tmp["cell_id"] = cba.index
    tmp.to_csv(output_path + "__" + str(crs_txt).replace(":", "_") + ".csv", index=False)


def to_raster(cba: gpd.GeoDataFrame, output_path: str) -> None:
    """Intermediate utility.

    Saves the object as a raster TIFF file.

    Args:
        output_tiff_path: name of the saved file.

    Returns:
        None
    """

    crs_txt = f"EPSG:{cba.crs.to_epsg()}"
    count = len(cba.columns.drop("geometry"))

    geometries = cba["geometry"].values
    x = np.unique(geometries.centroid.x)
    y = np.unique(geometries.centroid.y)
    X, Y = np.meshgrid(x, y)
    x_resolution = X[0][1] - X[0][0]
    y_resolution = Y[1][0] - Y[0][0]
    min_x, min_y, max_x, max_y = cba.total_bounds
    width = (max_x - min_x) / x_resolution
    height = (max_y - min_y) / y_resolution

    values = []
    col_name = []
    for col in cba.columns.drop("geometry"):
        col_name.append(col)
        val = cba[col].values
        val = val.reshape(X.shape)
        values.append(val)

    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width=width, height=height)

    with rasterio.open(
        output_path + ".tif",
        mode="w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=cba[col].dtype,
        crs=crs_txt,
        transform=transform,
        nodata=-9999,
    ) as new_dataset:
        for i in range(0, len(col_name)):
            z = i + 1
            new_dataset.write(values[i], z)
            new_dataset.set_band_description(z, col_name[i])
    new_dataset.close()
