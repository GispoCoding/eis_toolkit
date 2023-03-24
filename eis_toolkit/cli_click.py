"""Command-line interface for eis_toolkit."""
from pathlib import Path
from typing import Tuple

import click
import geopandas as gpd
import rasterio
from rasterio import warp

from eis_toolkit.raster_processing.clipping import clip_raster
from eis_toolkit.raster_processing.gridding_check import gridding_check
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.raster_processing.snapping import snap_with_raster
from eis_toolkit.raster_processing.windowing import extract_window

# from eis_toolkit.raster_processing.unifying import unify_rasters

EXISTING_CLICK_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


resampling_mapping = {
    "nearest": warp.Resampling.nearest,
    "bilinear": warp.Resampling.bilinear,
    "cubic": warp.Resampling.cubic,
    "average": warp.Resampling.average,
    "gauss": warp.Resampling.gauss,
    "max": warp.Resampling.max,
    "min": warp.Resampling.min,
}
# NOTE: Use the mapping below if all variants are wanted
# resampling_mapping = {method.name.lower(): method for method in warp.Resampling}


@click.group()
def cli():
    """Click group to nest subcommands under one interface."""
    pass


# CLIP RASTER
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("geometries", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True, help="Output raster file path")
def clip_raster_cli(input_raster: str, geometries: str, output_raster_file: str):
    """Clip the input raster with geometries in a geodataframe."""
    raster_path, geometries_path, output_raster_path = (
        Path(input_raster),
        Path(geometries),
        Path(output_raster_file),
    )

    geodataframe = gpd.read_file(geometries_path)
    with rasterio.open(raster_path) as raster:
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )

    click.echo("Clipping completed")
    click.echo(f"Writing raster to {output_raster_path}.")
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


# REPROJECT RASTER
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
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

    resampling = resampling_mapping[resampling_method.lower()]
    with rasterio.open(raster_path) as raster:
        out_image, out_meta = reproject_raster(src=raster, target_EPSG=crs, resampling_method=resampling)

    click.echo("Reprojecting completed")
    click.echo(f"Writing raster to {output_raster_path}.")
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


# SNAP WITH RASTER
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("snap_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True)
def snap_with_raster_cli(input_raster: str, snap_raster: str, output_raster_file: str):
    """Snaps/aligns input raster to the given snap raster."""
    with rasterio.open(input_raster) as src, rasterio.open(snap_raster) as snap_src:
        out_image, out_meta = snap_with_raster(src, snap_src)
    click.echo("Snapping completed")
    click.echo(f"Writing raster to {output_raster_file}")
    with rasterio.open(output_raster_file, "w", **out_meta) as dst:
        dst.write(out_image)


# EXTRACT WINDOW
@cli.command()
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
    click.echo("Windowing completed")
    click.echo(f"Writing raster to {output_raster_file}")
    with rasterio.open(output_raster_file, "w", **out_meta) as dst:
        dst.write(out_image)


# CHECK RASTER GRIDS
@cli.command()
@click.argument("input_rasters", nargs=-1, type=EXISTING_CLICK_FILE)
@click.option("--same-extent", type=bool, default=False)
def check_raster_grids_cli(input_rasters, same_extent: bool):
    """Check the set of input rasters for matching gridding and optionally matching bounds."""
    open_rasters = []
    for raster in input_rasters:
        open_rasters.append(rasterio.open(raster))
    check = gridding_check(open_rasters, same_extent)
    for raster in open_rasters:
        raster.close()
    click.echo(f"{check}")
    # click.echo(f"Gridding check returned {check}")
    return check


# PLACEHOLDER INTERFACES

# PCA
@cli.command()
@click.argument("input-raster", nargs=-1, type=EXISTING_CLICK_FILE)
@click.option("--components", type=int, required=True)
@click.option("--output-raster-file", type=click.Path(), required=True)
def pca_cli(input_raster, components: int, output_raster_file: str):
    """NOT IMPLEMENTED. Compute PCA with specified nr. of components."""
    raise Exception("Not implemented yet")


# FUZZY OVERLAY
@cli.command()
@click.argument("input_rasters", nargs=-1, type=EXISTING_CLICK_FILE)
@click.option("--fuzzy_method", type=str, required=True)
@click.option("--output-raster-file", type=click.Path(), required=True)
def fuzzy_overlay_cli(input_rasters, fuzzy_method: str, output_raster_file: str):
    """NOT IMPLEMENTED. Calculate fuzzy overlay."""
    raise Exception("Not implemented yet")


# GERETATE DATA
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True)
def generate_data_cli(input_raster: str, output_raster_file: str):
    """NOT IMPLEMENTED. Generate data for machine learning models."""
    raise Exception("Not implemented yet")


# RASTERIZE
@cli.command()
@click.argument("input_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--output-raster-file", type=click.Path(), required=True)
def rasterize_cli(input_raster: str, output_raster_file: str):
    """NOT IMPLEMENTED. Rasterize a given vector."""
    raise Exception("Not implemented yet")


# WEIGHTS OF EVIDENCE
@cli.command()
@click.argument("evidential_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("deposit_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--weights-type", type=str)
@click.option("--contrast", type=int)
@click.option("--output-raster-file", type=click.Path(), required=True)
def weights_of_evidence_cli(
    evidential_raster: str, deposit_raster: str, weights_type: str, contrast: int, output_raster_file: str
):
    """NOT IMPLEMENTED. Calculate weights of evidence."""
    raise Exception("Not implemented yet")


# UNIFY RASTERS
@cli.command()
@click.argument("base_raster", nargs=1, type=EXISTING_CLICK_FILE)
@click.argument("rasters", nargs=1, type=EXISTING_CLICK_FILE)
@click.option("--same-extent", type=bool, default=False)
@click.option("--output-raster-file", type=click.Path(), required=True)
def unify_rasters_cli(base_raster: str, rasters, same_extent: bool, output_raster_file: str):
    """NOT IMPLEMENTED. Unify given set of rasters."""
    raise Exception("Not implemented yet")


if __name__ == "__main__":
    cli()
