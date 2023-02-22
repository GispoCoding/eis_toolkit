import json
from pathlib import Path

import click
import geopandas as gpd
import jsonschema
import rasterio

from eis_toolkit.raster_processing.clipping import clip_raster

EXISTING_CLICK_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group()
def cli():
    """Click group to nest subcommands under one interface."""
    pass


def _cli_clip_raster(raster_path: Path, geodataframe_path: Path, output_raster_path: Path):
    click.echo(f"Reading geodataframe at {geodataframe_path}.")
    geodataframe = gpd.read_file(geodataframe_path)

    click.echo(f"Opening raster at {raster_path}.")
    with rasterio.open(raster_path) as raster:
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )
    click.echo(f"Writing raster to {output_raster_path}.")
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


@cli.command()
@click.argument("raster_file", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("geodataframe_file", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True)
def cli_clip_raster(raster_file: str, geodataframe_file: str, output_raster_file: str):
    """Clip a raster at ``raster_path`` with geometries in ``geodataframe_path``."""
    raster_path, geodataframe_path, output_raster_path = (
        Path(raster_file),
        Path(geodataframe_file),
        Path(output_raster_file),
    )
    _cli_clip_raster(
        raster_path=raster_path, geodataframe_path=geodataframe_path, output_raster_path=output_raster_path
    )


@cli.command()
@click.argument("config_file", nargs=1, type=EXISTING_CLICK_FILE)
def cli_clip_raster_config(config_file: str):
    """Clip a raster at raster_path with arguments from config_file.

    config_file should be a path to a valid JSON file with
    needed arguments for the clip_raster function.
    """
    assert isinstance(config_file, str)
    config_path = Path(config_file)
    loaded_config = json.loads(config_path.read_text())

    # I added an example on how to validate the configuration passed from a config file.
    # It uses the jsonschema library, which was, like click already a dependency
    schema = {
        "type": "object",
        "properties": {
            "raster_file": {
                "type": "string",
                "minLength": 1,
            },
            "geodataframe_file": {
                "type": "string",
                "minLength": 1,
            },
            "output_raster_file": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": ["raster_file", "geodataframe_file", "output_raster_file"],
    }

    jsonschema.validate(instance=loaded_config, schema=schema)

    click.echo(loaded_config)

    geodataframe_path = Path(loaded_config["geodataframe_file"])
    raster_path = Path(loaded_config["raster_file"])
    output_raster_path = Path(loaded_config["output_raster_file"])

    _cli_clip_raster(
        raster_path=raster_path, geodataframe_path=geodataframe_path, output_raster_path=output_raster_path
    )
