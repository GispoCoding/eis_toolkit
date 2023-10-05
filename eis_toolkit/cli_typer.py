import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
import typer
from rasterio import warp
from typing_extensions import Annotated

app = typer.Typer()


class VariogramModel(str, Enum):
    """Variogram models for kriging interpolation."""

    linear = "linear"
    power = "power"
    gaussian = "gaussian"
    spherical = "spherical"
    exponential = "exponential"


class CoordinatesType(str, Enum):
    """Coordinates type for kriging interpolation."""

    euclidean = "euclidean"
    geographic = "geographic"


class KrigingMethod(str, Enum):
    """Kriging methods."""

    ordinary = "ordinary"
    universal = "universal"


class MergeStrategy(str, Enum):
    """Merge strategies for rasterizing."""

    replace = "replace"
    add = "add"


class VectorDensityStatistic(str, Enum):
    """Vector density statistic."""

    density = "density"
    count = "count"


class LogarithmTransforms(str, Enum):
    """Logarithm choices for log transform."""

    ln = "ln"
    log2 = "log2"
    log10 = "log10"


class ResamplingMethods(str, Enum):
    """Resampling methods available."""

    nearest = "nearest"
    bilinear = "bilinear"
    cubic = "cubic"
    average = "average"
    gauss = "gauss"
    max = "max"
    min = "min"


RESAMPLING_MAPPING = {
    "nearest": warp.Resampling.nearest,
    "bilinear": warp.Resampling.bilinear,
    "cubic": warp.Resampling.cubic,
    "average": warp.Resampling.average,
    "gauss": warp.Resampling.gauss,
    "max": warp.Resampling.max,
    "min": warp.Resampling.min,
}


# def file_option(help: str = None, default: Any = None, read: bool = True, write: bool = False):
#     return typer.Option(
#         default=default,
#         help=help,
#         exists=True,
#         file_okay=True,
#         dir_okay=False,
#         writable=write,
#         readable=read,
#         resolve_path=True,
#     )


# TODO: Check this and output file option
INPUT_FILE_OPTION = typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    writable=False,
    readable=True,
    resolve_path=True,
)

OUTPUT_FILE_OPTION = typer.Option(
    file_okay=True,
    dir_okay=False,
    writable=True,
    readable=True,
    resolve_path=True,
)


# --- RASTER PROCESSING ---


# CHECK RASTER GRIDS
# TODO


# CLIP RASTER
@app.command()
def clip_raster_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    geometries: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
):
    """Clip the input raster with geometries in a geodataframe."""
    from eis_toolkit.raster_processing.clipping import clip_raster

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(geometries)

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Clipping completed, output raster written to {output_raster}.")


# CREATE CONSTANT RASTER
@app.command()
def create_constant_raster_cli(
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    constant_value: float = typer.Option(),
    template_raster: Annotated[Path, INPUT_FILE_OPTION] = None,
    coord_west: float = None,
    coord_north: float = None,
    coord_east: float = None,
    coord_south: float = None,
    target_epsg: int = None,
    target_pixel_size: int = None,
    raster_width: int = None,
    raster_height: int = None,
    nodata_value: float = None,
):
    """
    Create a constant raster with the given value.

    There are 3 methods for raster creation:
    - Set extent and coordinate system based on a template raster.
    - Set extent from origin, based on the western and northern coordinates and the pixel size.
    - Set extent from bounds, based on western, northern, eastern and southern points.
    """
    from eis_toolkit.raster_processing.create_constant_raster import create_constant_raster

    typer.echo("Progress: 10%")

    if template_raster is not None:
        with rasterio.open(template_raster) as raster:
            typer.echo("Progress: 25%")
            out_image, out_meta = create_constant_raster(
                constant_value,
                raster,
                coord_west,
                coord_north,
                coord_east,
                coord_south,
                target_epsg,
                target_pixel_size,
                raster_width,
                raster_height,
                nodata_value,
            )
    else:
        typer.echo("Progress: 25%")
        out_image, out_meta = create_constant_raster(
            constant_value,
            template_raster,
            coord_west,
            coord_north,
            coord_east,
            coord_south,
            target_epsg,
            target_pixel_size,
            raster_width,
            raster_height,
            nodata_value,
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        for band_n in range(1, out_meta["count"] + 1):
            dest.write(out_image, band_n)
    typer.echo("Progress: 100%")

    typer.echo(f"Creating constant raster completed, writing raster to {output_raster}.")


# EXTRACT VALUES FROM RASTER
@app.command()
def extract_values_from_raster_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    geometries: Annotated[Path, INPUT_FILE_OPTION],
    output_vector: Annotated[Path, OUTPUT_FILE_OPTION],
):
    """Extract raster values using point data to a DataFrame."""
    from eis_toolkit.raster_processing.extract_values_from_raster import extract_values_from_raster

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(geometries)

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        df = extract_values_from_raster(raster_list=[raster], geodataframe=geodataframe)
    typer.echo("Progress: 75%")

    df.to_csv(output_vector)
    typer.echo("Progress: 100%")

    typer.echo(f"Extracting values from raster completed, writing vector to {output_vector}.")


# REPROJECT RASTER
@app.command()
def reproject_raster_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    target_crs: int = typer.Option(help="crs help"),
    resampling_method: ResamplingMethods = typer.Option(help="resample help", default=ResamplingMethods.nearest),
):
    """Reproject the input raster to given CRS."""
    from eis_toolkit.raster_processing.reprojecting import reproject_raster

    typer.echo("Progress: 10%")

    method = RESAMPLING_MAPPING[resampling_method]
    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reproject_raster(raster=raster, target_crs=target_crs, resampling_method=method)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reprojecting completed, writing raster to {output_raster}.")


# SNAP RASTER
@app.command()
def snap_raster_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    snap_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
):
    """Snaps/aligns input raster to the given snap raster."""
    from eis_toolkit.raster_processing.snapping import snap_with_raster

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as src, rasterio.open(snap_raster) as snap_src:
        typer.echo("Progress: 25%")
        out_image, out_meta = snap_with_raster(src, snap_src)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Snapping completed, writing raster to {output_raster}.")


# UNIFY RASTERS
@app.command()
def unify_rasters_cli(
    base_raster: Annotated[Path, INPUT_FILE_OPTION],
    rasters_to_unify: Annotated[List[Path], INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    resampling_method: ResamplingMethods = typer.Option(help="resample help", default=ResamplingMethods.nearest),
    same_extent: bool = False,
):
    """Unify given rasters relative to base raster. NOT IMPLEMENTED YET."""
    # TODO
    pass


# EXTRACT WINDOW
@app.command()
def extract_window_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    center_coords: Tuple[float, float] = typer.Option(),
    height: int = typer.Option(),
    width: int = typer.Option(),
):
    """Extract window from raster."""
    from eis_toolkit.raster_processing.windowing import extract_window

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = extract_window(raster, center_coords, height, width)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Windowing completed, writing raster to {output_raster}")


# --- VECTOR PROCESSING ---


# IDW INTERPOLATION
@app.command()
def idw_interpolation_cli(
    input_vector: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    target_column: str = typer.Option(),
    resolution: float = typer.Option(),
    power: float = 2.0,
    extent: Tuple[float, float, float, float] = (None, None, None, None),  # TODO Change this
):
    """Apply inverse distance weighting (IDW) interpolation to input vector file."""
    from eis_toolkit.vector_processing.idw_interpolation import idw

    typer.echo("Progress: 10%")

    if extent == (None, None, None, None):
        extent = None

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    out_image, out_meta = idw(
        geodataframe=geodataframe,
        target_column=target_column,
        resolution=(resolution, resolution),
        extent=extent,
        power=power,
    )
    typer.echo("Progress: 75%")

    out_meta.update(
        {
            "count": 1,
            "driver": "GTiff",
            "dtype": "float32",
        }
    )

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"IDW interpolation completed, writing raster to {output_raster}.")


# KRIGING INTERPOLATION
@app.command()
def kriging_interpolation_cli(
    input_vector: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    target_column: str = typer.Option(),
    resolution: float = typer.Option(),
    extent: Tuple[float, float, float, float] = (None, None, None, None),  # TODO Change this
    variogram_model: VariogramModel = VariogramModel.linear,
    coordinates_type: CoordinatesType = CoordinatesType.geographic,
    method: KrigingMethod = KrigingMethod.ordinary,
):
    """Apply kriging interpolation to input vector file."""
    from eis_toolkit.vector_processing.kriging_interpolation import kriging

    typer.echo("Progress: 10%")

    if extent == (None, None, None, None):
        extent = None

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    out_image, out_meta = kriging(
        data=geodataframe,
        target_column=target_column,
        resolution=(resolution, resolution),
        extent=extent,
        variogram_model=variogram_model,
        coordinates_type=coordinates_type,
        method=method,
    )
    typer.echo("Progress: 75%")

    out_meta.update(
        {
            "count": 1,
            "driver": "GTiff",
            "dtype": "float32",
        }
    )

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Kriging interpolation completed, writing raster to {output_raster}.")


# RASTERIZE
@app.command()
def rasterize_cli(
    input_vector: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    resolution: float = None,
    value_column: str = None,
    default_value: float = 1.0,
    fill_value: float = 0.0,
    base_raster_profile_raster: Annotated[Path, INPUT_FILE_OPTION] = None,
    buffer_value: float = None,
    merge_strategy: MergeStrategy = MergeStrategy.replace,
):
    """
    Rasterize input vector.

    Either resolution or base-raster-profile-raster must be provided.
    """
    from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)

    if base_raster_profile_raster is not None:
        with rasterio.open(base_raster_profile_raster) as raster:
            base_raster_profile = raster.profile
    else:
        base_raster_profile = base_raster_profile_raster
    typer.echo("Progress: 25%")

    out_image, out_meta = rasterize_vector(
        geodataframe,
        resolution,
        value_column,
        default_value,
        fill_value,
        base_raster_profile,
        buffer_value,
        merge_strategy,
    )
    typer.echo("Progress: 75%")

    out_meta.update(
        {
            "count": 1,
            "dtype": base_raster_profile["dtype"] if base_raster_profile_raster else "float32",  # TODO change this
        }
    )

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        for band_n in range(1, out_meta["count"]):
            dst.write(out_image, band_n)
    typer.echo("Progress: 100%")

    typer.echo(f"Rasterizing completed, writing raster to {output_raster}.")


# REPROJECT VECTOR
@app.command()
def reproject_vector_cli(
    input_vector: Annotated[Path, INPUT_FILE_OPTION],
    output_vector: Annotated[Path, OUTPUT_FILE_OPTION],
    target_crs: int = typer.Option(help="crs help"),
):
    """Reproject the input vector to given CRS."""
    from eis_toolkit.vector_processing.reproject_vector import reproject_vector

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    reprojected_geodataframe = reproject_vector(geodataframe=geodataframe, target_crs=target_crs)
    typer.echo("Progress: 75%")

    reprojected_geodataframe.to_file(output_vector, driver="GeoJSON")
    typer.echo("Progress: 100%")

    typer.echo(f"Reprojecting completed, writing vector to {output_vector}.")


# VECTOR DENSITY
@app.command()
def vector_density_cli(
    input_vector: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    resolution: float = None,
    base_raster_profile_raster: Annotated[Path, INPUT_FILE_OPTION] = None,
    buffer_value: float = None,
    statistic: VectorDensityStatistic = VectorDensityStatistic.density,
):
    """
    Compute density of geometries within raster.

    Either resolution or base_raster_profile_raster must be provided.
    """
    from eis_toolkit.vector_processing.vector_density import vector_density

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)

    if base_raster_profile_raster is not None:
        with rasterio.open(base_raster_profile_raster) as raster:
            base_raster_profile = raster.profile
    else:
        base_raster_profile = base_raster_profile_raster
    typer.echo("Progress: 25%")

    out_image, out_meta = vector_density(
        geodataframe=geodataframe,
        resolution=resolution,
        base_raster_profile=base_raster_profile,
        buffer_value=buffer_value,
        statistic=statistic,
    )
    typer.echo("Progress: 75%")

    out_meta.update(
        {
            "count": 1,
            "dtype": base_raster_profile["dtype"] if base_raster_profile_raster else "float32",  # TODO change this
        }
    )

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        for band_n in range(1, out_meta["count"]):
            dst.write(out_image, band_n)
    typer.echo("Progress: 100%")

    typer.echo(f"Vector density computation completed, writing raster to {output_raster}.")


# --- SPATIAL ANALYSES ---


# DISTANCE COMPUTATION
@app.command()
def distance_computation_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    geometries: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
):
    """Calculate distance from raster cell to nearest geometry."""
    from eis_toolkit.spatial_analyses.distance_computation import distance_computation

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        profile = raster.profile

    geodataframe = gpd.read_file(geometries)
    typer.echo("Progress: 25%")

    out_image = distance_computation(profile, geodataframe)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, profile["count"])
    typer.echo("Progress: 100%")

    typer.echo(f"Distance computation completed, writing raster to {output_raster}.")


# CBA
# TODO


# --- STATISTICAL ANALYSES ---


# DESCRIPTIVE STATISTICS (RASTER)
@app.command()
def descriptive_statistics_raster_cli(input_file: Annotated[Path, INPUT_FILE_OPTION]):
    """Generate descriptive statistics from raster data."""
    from eis_toolkit.statistical_analyses.descriptive_statistics import descriptive_statistics_raster

    typer.echo("Progress: 10%")

    with rasterio.open(input_file) as raster:
        typer.echo("Progress: 25%")
        results_dict = descriptive_statistics_raster(raster)
    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")
    typer.echo("Descriptive statistics (raster) completed")


# DESCRIPTIVE STATISTICS (VECTOR)
@app.command()
def descriptive_statistics_vector_cli(input_file: Annotated[Path, INPUT_FILE_OPTION], column: str = None):
    """Generate descriptive statistics from vector or tabular data."""
    from eis_toolkit.statistical_analyses.descriptive_statistics import descriptive_statistics_dataframe

    typer.echo("Progress: 10%")

    # TODO modify input file detection
    try:
        gdf = gpd.read_file(input_file)
        typer.echo("Progress: 25%")
        results_dict = descriptive_statistics_dataframe(gdf, column)
    except:  # noqa: E722
        try:
            df = pd.read_csv(input_file)
            typer.echo("Progress: 25%")
            results_dict = descriptive_statistics_dataframe(df, column)
        except:  # noqa: E722
            raise Exception("Could not read input file as raster or dataframe")
    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 10%")

    typer.echo(f"Results: {json_str}")
    typer.echo("Descriptive statistics (vector) completed")


# --- PREDICTION ---


# FUZZY OVERLAY
# TODO


# WOFE
# TODO


# --- TRANSFORMATIONS ---


# BINARIZE
@app.command()
def binarize_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    threshold: float = typer.Option(),
):
    """
    Binarize data based on a given threshold.

    Replaces values less or equal threshold with 0.
    Replaces values greater than the threshold with 1.
    """
    from eis_toolkit.transformations.binarize import binarize

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = binarize(raster=raster, thresholds=[threshold])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Binarizing completed, writing raster to {output_raster}.")


# CLIP TRANSFORM
@app.command()
def clip_transform_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    limits: Tuple[Optional[float], Optional[float]] = typer.Option(),
):
    """
    Clips data based on specified upper and lower limits.

    Replaces values below the lower limit and above the upper limit with provided values, respecively.
    Works both one-sided and two-sided but raises error if no limits provided.
    """
    from eis_toolkit.transformations.clip import clip_transform

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = clip_transform(raster=raster, limits=limits)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Clip transform completed, writing raster to {output_raster}.")


# Z-SCORE NORMALIZATION
@app.command()
def z_score_normalization_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
):
    """
    Normalize data based on mean and standard deviation.

    Results will have a mean = 0 and standard deviation = 1.
    """
    from eis_toolkit.transformations.linear import z_score_normalization

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = z_score_normalization(raster=raster)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Z-score normalization completed, writing raster to {output_raster}.")


# MIX_MAX SCALING
@app.command()
def min_max_scaling_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    new_range: Tuple[float, float] = (0, 1),
):
    """
    Normalize data based on a specified new range.

    Uses the provided new minimum and maximum to transform data into the new interval.
    """
    from eis_toolkit.transformations.linear import min_max_scaling

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = min_max_scaling(raster=raster, new_range=[new_range])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Min-max scaling completed, writing raster to {output_raster}.")


# LOGARITHMIC
@app.command()
def log_transform_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    log_type: LogarithmTransforms = LogarithmTransforms.log2,
):
    """
    Perform a logarithmic transformation on the provided data.

    Logarithm base can be "ln", "log" or "log10".
    Negative values will not be considered for transformation and replaced by the specific nodata value.
    """
    from eis_toolkit.transformations.logarithmic import log_transform

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = log_transform(raster=raster, log_transform=[log_type])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Logarithm transform completed, writing raster to {output_raster}.")


# SIGMOID
@app.command()
def sigmoid_transform_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    bounds: Tuple[float, float] = (0, 1),
    slope: float = 1,
    center: bool = True,
):
    """
    Transform data into a sigmoid-shape based on a specified new range.

    Uses the provided new minimum and maximum, shift and slope parameters to transform the data.
    """
    from eis_toolkit.transformations.sigmoid import sigmoid_transform

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = sigmoid_transform(raster=raster, bounds=[bounds], slope=[slope], center=center)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Sigmoid transform completed, writing raster to {output_raster}.")


# WINSORIZE
@app.command()
def winsorize_transform_cli(
    input_raster: Annotated[Path, INPUT_FILE_OPTION],
    output_raster: Annotated[Path, OUTPUT_FILE_OPTION],
    percentiles: Tuple[Optional[float], Optional[float]] = typer.Option(),
    inside: bool = False,
):
    """
    Winsorize data based on specified percentile values.

    Replaces values between [minimum, lower percentile] and [upper percentile, maximum] if provided.
    Works both one-sided and two-sided but raises error if no percentile values provided.

    Percentiles are symmetrical, i.e. percentile_lower = 10 corresponds to the interval [min, 10%].
    And percentile_upper = 10 corresponds to the intervall [90%, max].
    I.e. percentile_lower = 0 refers to the minimum and percentile_upper = 0 to the data maximum.

    Calculation of percentiles is ambiguous. Users can choose whether to use the value
    for replacement from inside or outside of the respective interval. Example:
    Given the np.array[5 10 12 15 20 24 27 30 35] and percentiles(10, 10), the calculated
    percentiles are (5, 35) for inside and (10, 30) for outside.
    This results in [5 10 12 15 20 24 27 30 35] and [10 10 12 15 20 24 27 30 30], respectively.
    """
    from eis_toolkit.transformations.winsorize import winsorize

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = winsorize(raster=raster, percentiles=percentiles, inside=inside)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Winsorize transform completed, writing raster to {output_raster}.")


# ---VALIDATION ---
# TODO


# if __name__ == "__main__":
def cli():
    """CLI app."""
    app()


if __name__ == "__main__":
    app()
