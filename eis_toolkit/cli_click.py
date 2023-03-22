"""Command-line interface for eis_toolkit."""
from pathlib import Path
from typing import List, Tuple

import click
import geopandas as gpd
import rasterio
from rasterio import warp

from eis_toolkit.raster_processing.clipping import clip_raster
from eis_toolkit.raster_processing.gridding_check import gridding_check
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.raster_processing.snapping import snap_with_raster
from eis_toolkit.raster_processing.windowing import extract_window

EXISTING_CLICK_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group()
def cli():
    """Click group to nest subcommands under one interface."""
    pass


# CLIP RASTER
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE, help="Input raster to be clipped")
@click.argument("geodataframe", nargs=1, type=EXISTING_CLICK_FILE, help="Clipping geometries")
@click.option("--output-raster-file", type=click.Path(), required=True, help="Output raster file path")
def clip_raster_cli(input_raster: str, geodataframe: str, output_raster_file: str):
    """Clip the input raster with geometries in geodataframe."""
    raster_path, geodataframe_path, output_raster_path = (
        Path(input_raster),
        Path(geodataframe),
        Path(output_raster_file),
    )

    geodataframe = gpd.read_file(geodataframe_path)
    with rasterio.open(raster_path) as raster:
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )

    click.echo(f"Clipping completed")
    click.echo(f"Writing raster to {output_raster_path}.")
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


# REPROJECT RASTER
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE, help="Input raster to be reprojected")
@click.option("--crs", type=int, required=True, help="Target CRS as an EPSG code")
@click.option(
    "--output-raster-file", type=click.Path(), required=True, help="Path to the output reprojected raster file."
)
@click.option(
    "--resampling-method",
    type=click.Choice([m.name for m in warp.Resampling], case_sensitive=False),
    default=warp.Resampling.nearest.name,
    help="Resampling method to use. Defaults to 'nearest'.",
)
def reproject_raster_cli(input_raster: str, crs: str, output_raster_file: str, resampling_method: str):
    """Reproject the input raster to given coordinate refrence system."""
    raster_path, output_raster_path = (Path(input_raster), Path(output_raster_file))

    with rasterio.open(raster_path) as raster:
        out_image, out_meta = reproject_raster(src=raster, target_EPSG=crs, resampling_method=resampling_method)

    click.echo(f"Reprojecting completed")
    click.echo(f"Writing raster to {output_raster_path}.")
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


# SNAP WITH RASTER
@click.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("snap_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True)
def snap_with_raster_cli(input_raster: str, snap_raster: str, output_raster_file: str):
    """Snaps/aligns input raster to the given snap raster."""
    with rasterio.open(input_raster) as src, rasterio.open(snap_raster) as snap_src:
        out_image, out_meta = snap_with_raster(src, snap_src)
    click.echo(f"Snapping completed")
    click.echo(f"Writing raster to {output_raster_file}")
    with rasterio.open(output_raster_file, "w", **out_meta) as dst:
        dst.write(out_image)


# EXTRACT WINDOW
@click.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--center-coords", type=(float, float), required=True)
@click.option("--height", type=int, required=True)
@click.option("--width", type=int, required=True)
@click.option("--output-raster-file", type=click.Path(), required=True)
def extract_window_cli(
    input_raster: str, center_coords: Tuple[float, float], height: int, width: int, output_raster_file: str
):
    """Extract a window from a raster."""
    with rasterio.open(input_raster) as raster:
        out_image, out_meta = extract_window(raster, center_coords, height, width)
    click.echo(f"Windowing completed")
    click.echo(f"Writing raster to {output_raster_file}")
    with rasterio.open(output_raster_file, "w", **out_meta) as dst:
        dst.write(out_image)


# CHECK RASTER GRIDS
@click.command()
@click.argument("input-rasters", nargs=-1, type=EXISTING_CLICK_FILE)
@click.option("--same-extent", type=bool, default=False)
def check_matching_grids_cli(input_rasters, same_extent: bool):
    """Check the set of input rasters for matching gridding and optionally matching bounds."""
    open_rasters = []
    for raster in input_rasters:
        open_rasters.append(rasterio.open(raster))
    check = gridding_check(open_rasters, same_extent)
    for raster in open_rasters:
        raster.close()
    click.echo(f"Gridding check returned {check}")
