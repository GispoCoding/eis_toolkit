# -*- coding: utf-8 -*-

"""
Created on Thu Jan 19 09:17:33 2023.

@author: A.Vella, V. Labbe
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


class CBA:
    """
    Cell-Based Association class.

    Utility class allowing to transform one or more shapefile into a CSV file
    representing the projection of the specified attributes associated with the
    geometries of each shapefile on the CBA grid layout. For more details about
    the CBA, see for instance:

    Tourlière, B., Pakyuz-Charrier, E., Cassard, D., Barbanson, L., & Gumiaux,
    C. (2015). Cell Based Associations: A procedure for considering scarce and
    mixed mineral occurrences in predictive mapping. Computers & geosciences,
    78, 53-62.
    """

    def __init__(self) -> None:
        """Build an empty CBA matrix."""

        # cba is a GeoDataFrame object with the cell grid and the matrix.
        self.cba = gpd.GeoDataFrame()

        # crs is a string object (Coordinate system of the cba object).
        self.crs = str()

        # grid is a GeoDataFrame object containing only the cell grid.
        self.grid = gpd.GeoDataFrame()

    def init_from_vector_data(
        self,
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

        # Reset of the CBA grid (in case it has been previously initialized)
        self.cba = None

        # Reading the vector file
        geodata = geodataframe.copy(deep=True)
        self.crs = geodata.crs

        # Initialization of the grid
        grid = CBA._get_grid(geodata, cell_size, cell_size)
        grid.crs = geodata.crs
        self.grid = grid

        # Filling the grid with overlapped geometries
        indicators, join_grid = CBA._prepare_grid(geodataframe, grid, column, subset_of_target_attribute_values)
        tmp = join_grid[["geom"]].copy(deep=True)
        tmp.rename({"geom": "geometry"}, axis=1, inplace=True)

        # Saving results in attributes
        self.cba = gpd.GeoDataFrame(pd.concat([tmp.groupby(tmp.index).first(), indicators], axis=1))
        self.cba = self.cba.set_crs(self.crs, allow_override=True)
        self.cba = self.cba.astype("int", errors="ignore")

    def add_layer(
        self,
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
        assert self.cba is not None

        # Reading the vector file
        geodata = geodataframe.copy(deep=True)

        # No buffer
        if buffer is False:
            # Recovery of the grid calculated at the initialization of the CBA
            grid = gpd.GeoDataFrame(self.cba)

            # Adding a column to the CBA
            dummies, join_grid = CBA._prepare_grid(geodata, grid, column, subset_of_target_attribute_values)

        # Buffer of specified value
        else:
            # Calculation of the buffer disks
            added_buf = CBA._get_disc(self.cba, self.grid, buffer)

            # Adding a column to the CBA
            dummies, join_disc = CBA._prepare_disc(geodata, added_buf, column, subset_of_target_attribute_values)

            self.dummies = dummies
            self.join_disc = join_disc

        # Application of the indicated name (otherwise filename is used)
        if name is not None:
            dummies.name = name

        # Completion of the CBA object (values and names)
        self.cba = self.cba.join(dummies).replace(np.nan, 0)
        if column == "":
            self.cba[dummies.name] = self.cba[dummies.name].map(int)

        self.cba = self.cba.astype("int", errors="ignore")

    @staticmethod
    def _prepare_grid(  # type: ignore[no-any-unimported]
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
        dummification, target, identified_values = CBA._check_and_prepare_param(
            geodata, target_attribut, subset_of_target_attribut_values
        )

        # Spatial join of the intersection between the grid and the current map
        join_grid = gpd.sjoin(grid, geodata, how="inner", predicate="intersects")
        if dummification:
            tmp = pd.get_dummies(
                join_grid[join_grid[target].isin(identified_values)][[target]], prefix="", prefix_sep=""
            )
            indicators = tmp.groupby(tmp.index).max()
        else:
            tmp = join_grid[target]
            indicators = tmp.groupby(tmp.index).max()
        return indicators, join_grid

    @staticmethod
    def _check_and_prepare_param(  # type: ignore[no-any-unimported]
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

    @staticmethod
    def _get_disc(  # type: ignore[no-any-unimported]
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

    @staticmethod
    def _prepare_disc(  # type: ignore[no-any-unimported]
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
        dummification, target, identified_values = CBA._check_and_prepare_param(
            geodata, target_attribut, subset_of_target_attribut_values
        )

        # Spatial join of the intersection between the grid and the current map
        join_grid = gpd.sjoin(disc, geodata, how="inner", predicate="intersects")

        if dummification:
            tmp = pd.get_dummies(
                join_grid[join_grid[target].isin(identified_values)][[target]], prefix="", prefix_sep=""
            )
            indicators = tmp.groupby(tmp.index).max()
        else:
            tmp = join_grid[target]
            indicators = tmp.groupby(tmp.index).max()
        return indicators, join_grid

    @staticmethod
    def _get_grid(  # type: ignore[no-any-unimported]
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

    @staticmethod
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
        cba_grid = CBA()
#        crs = {"init": tmp[1].split(".")[0].replace("_", ":")}
        crs = tmp[1].split(".")[0].replace("_", ":")
        df = pd.read_csv(input_csv_file_path)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        cba_grid.cba = gpd.GeoDataFrame(df, geometry="geometry")
        cba_grid.cba = cba_grid.cba.set_crs(crs)
        cba_grid.crs = cba_grid.cba.crs
        cba_grid.grid = cba_grid.cba[["geometry"]]
        cba_grid.grid.index = cba_grid.cba.index
        return cba_grid

    @staticmethod
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
        cba_grid = CBA()
        geodata = gpd.GeoDataFrame.from_file(cba_vector_file_path)
        cba_grid.cba = gpd.GeoDataFrame(geodata, geometry="geometry")
        cba_grid.cba.set_index("cell_id", inplace=True)
        cba_grid.crs = geodata.crs
        cba_grid.grid = geodata[["geometry"]]
        cba_grid.grid.index = cba_grid.cba.index
        cols = cba_grid.cba.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        cba_grid.cba = cba_grid.cba[cols]
        return cba_grid

    def to_csv(self, output_csv_file_path: str) -> None:
        """Intermediate utility.

        Saves the object in a CSV file.

        Args:
            output_csv_file_path: name of the saved file suffixed with the
                CRS (spatial projection system).

        Returns:
            None
        """

        crs_txt = f"epsg:{self.cba.crs.to_epsg()}"
        tmp = gpd.GeoDataFrame(self.cba)
        tmp.to_csv(output_csv_file_path + "__" + str(crs_txt).replace(":", "_") + ".csv", index=False)

    def to_shapefile(self, output_shapefile_path: str) -> None:
        """Intermediate utility.

        Saves the object in an ESRI SHP file.

        Args:
            output_shapefile_path: name of the saved file.

        Returns:
            None
        """

        tmp = gpd.GeoDataFrame(self.cba)
        tmp.crs = self.crs
        tmp.to_file(output_shapefile_path + ".shp")

    def to_geojson(self, output_geojson_file_path: str) -> None:
        """Intermediate utility.

        Saves the object in a geoJSON file.

        Args:
            output_geojson_file_path: name of the saved file.

        Returns:
            None
        """

        tmp = gpd.GeoDataFrame(self.cba)
        tmp.crs = self.crs
        tmp.to_file(output_geojson_file_path + ".geojson", driver="GeoJSON")

    def to_raster(self, output_tiff_path: str, close_at_end: bool=True) -> None:
        """Intermediate utility.

        Saves the object as a raster TIFF file.

        Args:
            output_tiff_path: name of the saved file.

        Returns:
            None
        """

        crs_txt = f"EPSG:{self.cba.crs.to_epsg()}"
        count = len(self.cba.columns.drop("geometry"))

        geometries = self.cba["geometry"].values
        x = np.unique(geometries.centroid.x)
        y = np.unique(geometries.centroid.y)
        X, Y = np.meshgrid(x, y)
        x_resolution = X[0][1] - X[0][0]
        y_resolution = Y[1][0] - Y[0][0]
        min_x, min_y, max_x, max_y = self.cba.total_bounds
        width = (max_x - min_x) / x_resolution
        height = (max_y - min_y) / y_resolution

        values = []
        col_name = []
        for col in self.cba.columns.drop("geometry"):
            col_name.append(col)
            val = self.cba[col].values
            val = val.reshape(X.shape)
            values.append(val)

        transform  = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width=width, height=height)

        with rasterio.open(
                output_tiff_path + ".tif",
                mode="w",
                driver="GTiff",
                height=height,
                width=width,
                count=count,
                dtype=self.cba[col].dtype,
                crs=crs_txt,
                transform=transform,
                nodata=-9999
        ) as new_dataset:
                for i in range(0, len(col_name)):
                    z = i + 1
                    new_dataset.write(values[i], z)
                    new_dataset.set_band_description(z, col_name[i])
        if close_at_end == True:
            new_dataset.close()
