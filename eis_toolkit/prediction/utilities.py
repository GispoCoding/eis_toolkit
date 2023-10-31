from typing import Any

import numpy as np
from beartype import beartype
from osgeo import gdal


@beartype
def parse_the_master_file(master_file_path) -> dict:
    """
    Load sets of windows from its folder.

    Parameters:
        master_file_path: path of the masterfile.

    Return:
        Dictionary that contains the following information:
                current_path: this the path of the feature.
                no_data: no data value.
                no_data_value: which value to substitute to the no data.
                min_range_value: min range of the raster.
                max_range_value: max range of the.
                channel: band.
                windows_dimension: the window dimension.
                valid_bands: valid band to use.
                loaded_dataset_as_array: current raster array.
                current_geo_transform: current set of N and E.
                model_type: this is not used here.
    Raises:
        TODO
    """
    current_dataset = dict()
    # open the handler
    handler = open(master_file_path)

    # loop inside each rows
    for line in handler.readlines():
        line_to_split = line.strip()

        # get band list and convert to int
        bands = line_to_split.split(":")[1].split(",")

        # parse features to int
        bands = [int(x) for x in bands]

        # get other values
        others_values = line_to_split.split(":")[0].split(",")

        # che sub raster from raster
        loaded_raster = gdal.Open(f"{others_values[0]}", gdal.GA_ReadOnly)
        geo = loaded_raster.GetGeoTransform()

        # create a key holder for the dict of feat
        if others_values[1] not in current_dataset.keys():
            current_dataset[others_values[1]] = list()

            current_dataset[others_values[1]].append(
                {
                    "current_path": f"{others_values[0].split('/')[-2]}/{others_values[0].split('/')[-1]}",
                    "no_data": float(others_values[3]) if others_values[3] != "" else "",
                    "no_data_value": float(others_values[4]) if others_values[4] != "" else 255,
                    "min_range_value": float(others_values[5]) if others_values[5] != "" else "",
                    "max_range_value": float(others_values[6]) if others_values[6] != "" else "",
                    "channel": others_values[1],
                    "windows_dimension": int(others_values[2]),
                    "valid_bands": bands,
                    "loaded_dataset_as_array": loaded_raster.ReadAsArray()[bands, :, :]
                    if loaded_raster.ReadAsArray().ndim > 2
                    else loaded_raster.ReadAsArray(),
                    "current_geo_transform": geo,
                    "model_type": others_values[7],
                }
            )
    handler.close()
    return current_dataset


@beartype
def return_list_of_N_and_E(path_to_data: str) -> list[list[Any]]:
    """
    Load the list of N and E coordinates.

    Parameters:
        input_path: this is what in keras is called optimizer.

    Return:
        a list that contain all N end E

    Raises:
        TODO
    """
    # load the csv file with the deposit annotation
    handler = open(path_to_data, "r")
    coords = list()
    for row_counter, row in enumerate(handler.readlines()):
        if row_counter != 0:
            coords.append([row.strip().split(",")[-2], row.strip().split(",")[-3]])
    return coords


@beartype
def create_windows_based_of_geo_coords(
    current_raster_object: dict,
    current_E: float,
    current_N: float,
    desired_windows_dimension: int,
    current_loaded_raster: gdal.Dataset,
) -> np.ndarray:
    """
    Create windows from geo coordinates.

    Parameters:
       current_raster_object: raster iformation from the masterfile.
       current_E: float point showing the E.
       current_N: float point showing the N.
       desired_windows_dimension: int dimension of the window.
       current_loaded_raster: load tif with gdal

    Return:
        numpy array with the windows inside.

    Raises:
        TODO
    """

    if current_loaded_raster is not None:
        # get the loaded raster and the pix
        current_raster = gdal.Open(current_raster_object["current_path"])
    else:
        current_raster = current_loaded_raster

    spatial_pixel_resolution = current_raster_object["current_geo_transform"][1]

    # get the coords
    start_N = current_N + (desired_windows_dimension / 2) * spatial_pixel_resolution
    end_N = start_N + desired_windows_dimension * spatial_pixel_resolution

    start_E = current_E - (desired_windows_dimension / 2) * spatial_pixel_resolution
    end_E = start_E + desired_windows_dimension * spatial_pixel_resolution

    raster = gdal.Warp(
        "",
        current_raster,
        outputBounds=[start_E, end_N, end_E, start_N],
        format="MEM",
        xRes=spatial_pixel_resolution,
        yRes=-spatial_pixel_resolution,
    )

    values_with_need = raster.ReadAsArray()

    # create the window I m testing with float
    window = (
        np.array(values_with_need).astype("float32").reshape((desired_windows_dimension, desired_windows_dimension, -1))
    )

    # remove no data value:
    if current_raster_object["no_data"] != "":
        window[window == current_raster_object["no_data"]] = current_raster_object["no_data_value"]

    return window
