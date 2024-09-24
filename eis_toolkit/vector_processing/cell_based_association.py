import os
import warnings
from numbers import Number
from os import PathLike

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from shapely import wkt
from shapely.geometry import Point, Polygon

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidColumnException, InvalidParameterValueException

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ["USE_PYGEOS"] = "0"


@beartype
def cell_based_association(
    cell_size: int,
    geodata: List[gpd.GeoDataFrame],
    output_path: Union[str, PathLike],
    column: Optional[List[str]] = None,
    subset_target_attribute_values: Optional[List[Union[None, list, str]]] = None,
    add_name: Optional[List[Union[str, None]]] = None,
    add_buffer: Optional[List[Union[Number, bool]]] = None,
) -> gpd.GeoDataFrame:
    """Creation of CBA matrix.

    Initializes a CBA matrix from a vector file. The mesh is calculated
    according to the geometries contained in this file and the size of cells.
    Allows to add multiple vector data to the matrix, based on targeted shapes
    and/or attributes.

    Args:
        cell_size: Size of the cells.
        geodata: GeoDataFrame to create the CBA matrix. Additional
            GeoDataFrame(s) can be imputed to add to the CBA matrix.
        output_path: Name of the saved .tif file. Include file extension (.tif)
            in the path.
        column: Name of the column of interest. If no attribute is specified,
            then an artificial attribute is created representing the presence
            or absence of the geometries of this file for each cell of the CBA
            grid. A categorical attribute will generate as many columns (binary)
            in the CBA matrix than values considered of interest (dummification).
            See parameter <subset_target_attribute_values>. Additional
            column(s) can be imputed for each added GeoDataFrame(s).
        subset_target_attribute_values: List of values of interest of the
            target attribute, in case a categorical target attribute has been
            specified. Allows to filter a subset of relevant values. Additional
            values can be imputed for each added GeoDataFrame(s).
        add_name: Name of the column(s) to add to the matrix.
        add_buffer: Allow the use of a buffer around shapes before the
            intersection with CBA cells for the added GeoDataFrame(s). Minimize
            border effects or allow increasing positive samples (i.e. cells
            with mineralization). The size of the buffer is computed using the
            CRS (if projected CRS in meters: value in meters).

    Returns:
        CBA matrix is created.
    """

    # Swapping None to list values
    if column is None:
        column = [""]
    if add_buffer is None:
        add_buffer = [False]

    # Consistency checks on input data
    for frame in geodata:
        if frame.empty:
            raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if cell_size <= 0:
        raise InvalidParameterValueException("Expected cell size to be positive and non-zero.")

    add_buffer = [False if x == 0 else x for x in add_buffer]
    if any(num < 0 for num in add_buffer):
        raise InvalidParameterValueException("Expected buffer value to be positive, null or False.")

    for i, name in enumerate(column):
        if column[i] == "":
            if subset_target_attribute_values[i] is not None:
                raise InvalidParameterValueException("Can't use subset of values if no column is targeted.")
        elif column[i] not in geodata[i]:
            raise InvalidColumnException("Targeted column not found in the GeoDataFrame.")

    for i, subset in enumerate(subset_target_attribute_values):
        if subset is not None and not all(value in geodata[i][column[i]].tolist() for value in subset):
            raise InvalidParameterValueException("Subset of value(s) not found in the targeted column.")

    # Computation
    for i, data in enumerate(geodata):
        if i == 0:
            # Initialization of the CBA matrix
            grid, cba = _init_from_vector_data(cell_size, geodata[0], column[0], subset_target_attribute_values[0])
        else:
            # If necessary, adding data to matrix
            cba = _add_layer(
                cba,
                grid,
                geodata[i],
                column[i],
                subset_target_attribute_values[i],
                add_name[i - 1],
                add_buffer[i - 1],
            )

    # Export
    _to_raster(cba, output_path)

    return cba


@beartype
def _init_from_vector_data(
    cell_size: int,
    geodataframe: gpd.GeoDataFrame,
    column: str = "",
    subset_target_attribute_values: Optional[list] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Creation of CBA matrix.

    Initializes a CBA matrix from a vector file. The mesh is calculated
    according to the geometries contained in this file and the size
    of cells.

    Args:
        cell_size: Size of the cells.
        geodataframe: GeoDataframe to create the CBA matrix.
        column: Name of the column of interest.
            If no attribute is specified, then an artificial attribute is
            created representing the presence or absence of the geometries of
            this file for each cell of the CBA grid. Categorical attribute will
            generate as many columns (binary) in the CBA matrix than values
            considered of interest(dummification). See parameter
            <subset_target_attribute_values>
        subset_target_attribute_values: List of values of interest of the
            target column, in case a categorical target attribute has been
            specified. Allows to filter a subset of relevant values. If set to
            an empty list, each unique value in the targeted column is dummified.

    Returns:
        Tuple of GeodataFrames: the grid mesh produced and the CBA matrix.
    """

    # Allows a non-mutable default value
    if subset_target_attribute_values is None:
        subset_target_attribute_values = []

    # Reading the vector file
    geodata = geodataframe.copy(deep=True)

    # Converting the targeted column to categorical dtype
    if column != "":
        geodata = geodata.astype({column: "category"})

    # Initialization of the grid
    grid = _get_grid(geodata, cell_size, cell_size)

    # Filling the grid with overlapped geometries
    indicators, join_grid = _prepare_grid(geodataframe, grid, column, subset_target_attribute_values)
    tmp = join_grid[["geometry"]].copy(deep=True)

    # Saving results in attributes
    cba = gpd.GeoDataFrame(pd.concat([tmp.groupby(tmp.index).first(), indicators], axis=1))
    cba = cba.set_crs(grid.crs, allow_override=True)
    cba = cba.astype("int32", errors="ignore")

    return grid, cba


@beartype
def _add_layer(
    cba: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    geodataframe: gpd.GeoDataFrame,
    column: str = "",
    subset_target_attribute_values: Optional[Union[list, str]] = None,
    name: Optional[str] = None,
    buffer: Union[Number, bool] = False,
) -> gpd.GeoDataFrame:
    """Add new data to CBA matrix.

    Allow to add data to an existing matrix. Mesh size and placement are not
    altered. On or multiple columns are added to the CBA matrix based on
    targeted shapes and/or attributes.

    Args:
        cba: Matrix produced when initializing the CBA.
        grid: Grid mesh produced when initializing the CBA.
        geodataframe: GeodataFrame to add to the CBA matrix.
        column: Name of the targeted attribute. If
            no attribute are specified, added data fo each cell are based on
            the absence/presence intersection between cells and the shapes
            within the selected vector file. If specified, attribute must be
            categorical (list) and will generate as much columns as it has
            unique values in <subset_target_attribute_values>
            (dummification).
            Note: if a file contain multiple target attribute, <add_layer> has
            to be executed as many times as there are target attributes in this
            file.
        subset_target_attribute_values: List of categorical values of
            interest within the targeted column. Allow filtering subset of
            relevant values.
        name: Name of the column to add to the matrix.
        buffer: Allow the use of a buffer around shapes before
            the intersection with CBA cells.Minimize border effects or allow
            increasing positive samples (i.e. cells with mineralization).
            The size of the buffer is computed using the CRS (if projected CRS
            in meters: value in meters).

    Returns:
        Data added to the CBA matrix.
    """

    # Allows a non-mutable default value
    if subset_target_attribute_values is None:
        subset_target_attribute_values = []

    # Reading the vector file
    geodata = geodataframe.copy(deep=True)

    # Converting the targeted column to categorical dtype
    if column != "":
        geodata = geodata.astype({column: "category"})

    # No buffer
    if buffer is False:
        # Recovery of the grid calculated at the initialization of the CBA
        grid = gpd.GeoDataFrame(cba.copy())

        # Adding a column to the CBA
        dummies, join_grid = _prepare_grid(geodata, grid, column, subset_target_attribute_values)

    # Buffer of specified value
    else:
        # Calculation of the buffer disks
        added_buf = _get_disc(cba, grid, buffer)

        # Adding a column to the CBA
        dummies, join_disc = _prepare_grid(geodata, added_buf, column, subset_target_attribute_values)

    # Application of the indicated name (otherwise filename is used)
    if name is not None:
        dummies.name = name
        if column != "":
            dummies.columns = [str(name) + str(col) for col in dummies.columns]

    # Completion of the CBA object (values and names)
    cba = cba.join(dummies).replace(np.nan, 0)
    if column == "":
        cba[dummies.name] = cba[dummies.name].map(int)

    cba = cba.astype("int32", errors="ignore")

    return cba


@beartype
def _prepare_grid(
    geodata: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    column: str,
    subset_target_attribute_values: Union[list, str],
) -> Tuple[Union[pd.DataFrame, pd.Series], gpd.GeoDataFrame]:
    """Intermediate utility.

    Preparation of the dummification of selected variables and intersection
    between grid and input dataset.

    Args:
        geodata: The geodata object corresponding to the vector file.
        grid: Grid mesh produced when initializing the CBA.
        column: Column attribute to extract from the vector file.
        subset_target_attribute_values: Subset of the relevant values
            of the target column.

    Returns: pair of variables
        -> Boolean indicating if the target attribute must be dummified
        -> Grid resulting from the intersection between the mesh and the
        vector file
    """

    # Data verification
    dummification, target, identified_values = _check_and_prepare_param(geodata, column, subset_target_attribute_values)

    # Spatial join of the intersection between the grid and the current map
    join_grid = gpd.sjoin(grid, geodata, how="inner", predicate="intersects")
    if dummification:
        tmp = pd.get_dummies(join_grid[join_grid[target].isin(identified_values)][[target]], prefix="", prefix_sep="")
        indicators = tmp[[str(x) for x in identified_values]].groupby(tmp.index).max()
    else:
        tmp = join_grid[target]
        indicators = tmp.groupby(tmp.index).max()
    return indicators, join_grid


@beartype
def _check_and_prepare_param(
    geodata: gpd.GeoDataFrame,
    column: str,
    subset_target_attribute_values: Union[list, str],
    target_name: Optional[str] = "Added",
) -> Tuple[bool, str, list]:
    """Intermediate utility.

    Prepare the parameters and check their consistency.

    Args:
        geodata: Geodata object corresponding to the vector file
        column: Name of the column of interest. If no attribute is
            specified, a synthetic one is produced to represent the
            absence/presence of geometry from this file within each cell of the
            CBA. If an attribute is specified, it must be categorical. In this
            case, it will produce as much columns (binary) in the CBA matrix
            than listed values. See parameter
            <subset_target_attribute_values>. Note: if a vector file contains
            several target attributes, it is necessary to run as many times the
            <add_layer> function than target attributes of this file.
        subset_target_attribute_values: List of values of interest for the
            target column, in the case where a specific categorical target
            has been specified. It allows filtering a subset of relevant
            values.
        target_name: Name of the produced boolean column when using all
            geometries to dummify the data.

    Returns: triplet
        -> dummification: Boolean indicating the dummification (processing
            of categorical attribute)
        -> target: Targeted variable, can be artificial when only geometries
            are considered.
        -> identified_values: Subset of values of interest, possibly artificial
            when only geometries are considered.
    """

    dummification = False
    if column == "":
        # Case where we use geometries (e.g. a point file for mineralization)
        # we add a column filled with 1 (presence).
        column = target_name
        subset_target_attribute_values = [1]
        geodata[column] = int(1)
    else:
        # Case of a symbolic attribute with identified subset of values of
        # interest.
        dummification = True
        # Case when selecting all attribute values in the targeted column
        if subset_target_attribute_values == []:
            subset_target_attribute_values = geodata[column].unique()
    identified_values = pd.Series(subset_target_attribute_values)
    if column not in geodata.columns:
        raise InvalidColumnException("Targeted column not found in the GeoData.")
    if identified_values.isin(geodata[column]).all() is False:
        raise InvalidParameterValueException("Subset of value(s) not found in the targeted column.")
    if len(subset_target_attribute_values) == 0:
        raise InvalidParameterValueException("Subset of value(s) is empty.")

    return dummification, column, list(identified_values)


@beartype
def _get_disc(geodata: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, buff_radius: Union[int, float]) -> gpd.GeoDataFrame:
    """Intermediate utility.

    Defines disks with radii defined from the centroids of the cells of a grid.

    Args:
        geodata: Geodata to intersect with the grid.
        grid: Grid from which the discs will be created.
        buff_radius: Radius for created buffer discs

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
    disc["geometry"] = disc.geometry.buffer(buff_radius, resolution=16)

    return gpd.GeoDataFrame(disc, geometry="geometry", crs=geodata.crs)


@beartype
def _get_grid(working_map: gpd.GeoDataFrame, cell_width: int, cell_height: int) -> gpd.GeoDataFrame:
    """Intermediate utility.

    Produces a regular grid based on geodata surface.
    Paving begin from the upper left corner (North-West) and go to the South
    and East (up to down, left to right paving).

    Args:
        working_map: Geodata used as a 'bounding box'.
        cell_width: Width of cells.
        cell_height: Height of cells.

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
    result = pd.DataFrame(data, columns=["cell_id", "x", "y", "geometry"])
    result.set_index("cell_id", inplace=True)

    return gpd.GeoDataFrame(result, geometry="geometry", crs=working_map.crs)


@beartype
def _from_csv(input_csv_file_path: str) -> gpd.GeoDataFrame:
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
    df["geometry"] = df["geometry"].apply(wkt.loads)
    # df.insert(0, "geometry", df["geometry"].apply(wkt.loads))
    # df["geometry"] = df["geom"].apply(wkt.loads)
    # df = df.drop("geom", axis=1)
    cba_grid = gpd.GeoDataFrame(df, geometry="geometry")
    cba_grid = cba_grid.set_crs(crs)
    cba_grid.index = df["cell_id"]
    cba_grid = cba_grid.drop("cell_id", axis=1)
    cba_grid = cba_grid.astype("int32", errors="ignore")
    return cba_grid


@beartype
def _from_vector_file(cba_vector_file_path: str) -> gpd.GeoDataFrame:
    """Intermediate utility.

    Read a CBA grid from a vector file (ESRI shapefile, geojson...).
    All attributes (and all values of each attribute) are kept.
    The geometries contained in the file are assumed to represent a regular
    square grid.

    Args:
        cba_vector_file_path: Input vector file.

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
    cba_grid = cba_grid.astype("int32", errors="ignore")
    return cba_grid


@beartype
def _to_csv(cba: gpd.GeoDataFrame, output_path: str) -> None:
    """Intermediate utility.

    Saves the object in a CSV file.

    Args:
        cba: CBA matrix to save.
        output_path: Name of the saved file suffixed with the
            CRS (spatial projection system).

    Returns:
        None
    """

    crs_txt = f"epsg:{cba.crs.to_epsg()}"
    tmp = pd.DataFrame(cba.copy())
    tmp["geometry"] = tmp.geometry.apply(lambda x: wkt.dumps(x))
    # tmp = tmp.drop("geometry", axis=1)
    tmp["cell_id"] = cba.index
    tmp.to_csv(output_path + "__" + str(crs_txt).replace(":", "_") + ".csv", index=False)


@beartype
def _to_raster(cba: gpd.GeoDataFrame, output_path: Union[str, PathLike], nan_val: int = -9999) -> None:
    """Intermediate utility.

    Saves the object as a raster TIFF file.

    Args:
        cba: CBA matrix to save.
        output_path: Name of the saved file, include file extension (.tif).
        nan_val: values taken by cells with no values in them (outside the study
        area).

    Returns:
        None
    """

    cba = cba.copy(deep=True)

    crs_txt = f"EPSG:{cba.crs.to_epsg()}"
    count = len(cba.columns.drop("geometry"))

    geometries = cba["geometry"].values
    x = np.unique(geometries.centroid.x)
    y = np.unique(geometries.centroid.y)
    y = np.flip(y)
    x_resolution = x[1] - x[0]
    y_resolution = y[0] - y[1]
    min_x, min_y, max_x, max_y = cba.total_bounds
    width = round((max_x - min_x) / x_resolution)
    height = round((max_y - min_y) / y_resolution)

    x_values = [x[0]]
    for i in range(0, (width - 1), 1):
        new_val = x_values[i] + x_resolution
        x_values = np.append(x_values, new_val)
    y_values = [y[0]]
    for i in range(0, (height - 1), 1):
        new_val = y_values[i] - y_resolution
        y_values = np.append(y_values, new_val)
    X, Y = np.meshgrid(x_values, y_values)

    points = pd.DataFrame({"X": np.ravel(X), "Y": np.ravel(Y)})
    points["coords"] = list(zip(points["X"], points["Y"]))
    points["coords"] = points["coords"].apply(Point)
    points_grid = gpd.GeoDataFrame(points, geometry="coords", crs=cba.crs)
    points_grid = points_grid.sjoin(cba, how="left")
    points_grid = points_grid.fillna(nan_val)
    col_name = list(points_grid.drop(["X", "Y", "coords", "index_right"], axis=1).columns)

    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width=width, height=height)

    with rasterio.open(
        output_path,
        mode="w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="int32",
        crs=crs_txt,
        transform=transform,
        nodata=nan_val,
    ) as new_dataset:
        z = 1
        for i in col_name:
            new_dataset.write(points_grid.pivot(index="Y", columns="X", values=i).sort_index(ascending=False).values, z)
            new_dataset.set_band_description(z, i)
            z = z + 1
    new_dataset.close()
