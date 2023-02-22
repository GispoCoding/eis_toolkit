from pathlib import Path

import click
import geopandas as gpd
import rasterio

from eis_toolkit.raster_processing.clipping import clip_raster

EXISTING_CLICK_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group()
def cli():
    """Click group to nest subcommands under one interface."""
    pass


@cli.command()
@click.argument("raster_path", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("geodataframe_path", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-path", type=click.Path(), required=True)
def cli_clip_raster(raster_path: Path, geodataframe_path: Path, output_raster_path: Path):
    """Clip a raster at ``raster_path`` with geometries in ``geodataframe_path``."""

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
