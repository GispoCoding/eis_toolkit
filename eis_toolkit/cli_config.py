import json
from pathlib import Path

import geopandas as gpd
import rasterio


def convert_arguments(arg_data):
    function_arguments = {}
    for arg, arg_items in arg_data:
        arg_type = arg_items["type"]
        arg_value = arg_items["value"]
        converted_arg = convert_argument(arg_value, arg_type)
        function_arguments[arg] = converted_arg


def convert_argument(arg_value, arg_type):
    if arg_type == "raster":
        converted_arg = rasterio.open(arg_value)
    elif arg_type == "geodataframe":
        converted_arg = gpd.read_file(arg_value)
    else:
        raise Exception(f"Unknown argument type (arg type, arg value): {arg_type, arg_value}")

    return converted_arg


def write_outputs(output_definitions, output_data):
    for arg, arg_items in output_definitions:
        arg_type = arg_items["type"]
        arg_value = arg_items["value"]
        write_output(arg_value, arg_type, output_data)


def write_output(arg_value, arg_type, data):
    if arg_type == "raster":
        out_image = data[0]
        out_meta = data[1]
        with rasterio.open(arg_value, "w", **out_meta) as dest:
            dest.write(out_image)
    elif arg_type == "vector":
        pass


def read_config(config_path):
    config_path = Path(config_path)
    loaded_config = json.loads(config_path.read_text())


def execute_workflow(config):
    for function_name, func_items in config["workflow"].items():
        function = locals()[function_name]
        function_arguments = convert_arguments(func_items["inputs"].items())
        output_data = function(**function_arguments)
        write_outputs(func_items["outputs"].items(), output_data)


def interface():
    config_path = "/home/niko/code/plugin_dev/eis_toolkit/eis_toolkit/config_test.json"
    loaded_config = read_config(config_path)
    execute_workflow(loaded_config)


interface()
