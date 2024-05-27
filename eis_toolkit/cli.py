# --- ! ---
# NOTE! Work in progress in the implementation of command-line interface
# Note also, that this CLI is primarily created for other applications to
# utilize EIS Toolkit, such as EIS QGIS Plugin
# --- ! ---

import json
import os
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import typer
from beartype.typing import List, Optional, Tuple, Union
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


class AngleUnits(str, Enum):
    """Unit for classify aspect."""

    radians = "radians"
    degrees = "degrees"


class SlopeGradientUnit(str, Enum):
    """Slope gradient unit for first order surface derivatives."""

    degrees = "degrees"
    radians = "radians"
    rise = "rise"


class FirstOrderMethod(str, Enum):
    """Method for first order surface derivatives."""

    Horn = "Horn"
    Evans = "Evans"
    Young = "Young"
    Zevenbergen = "Zevenbergen"


class SurfaceParameter(str, Enum):
    """Parameter choice for surface derivatives."""

    G = "G"
    A = "A"
    planc = "planc"
    profc = "profc"
    profc_min = "profc_min"
    profc_max = "profc_max"
    longc = "longc"
    crosc = "crosc"
    rot = "rot"
    K = "K"
    genc = "genc"
    tangc = "tangc"


class SecondOrderMethod(str, Enum):
    """Method for second order surface derivatives."""

    Evans = "Evans"
    Young = "Young"
    Zevenbergen = "Zevenbergen"


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


class ValidationMethods(str, Enum):
    """Validation methods available."""

    split_once = "split"  # Note that split_once might be the new name
    kfold_cv = "kfold_cv"
    skfold_cv = "skfold_cv"
    loo_cv = "loo_cv"
    none = "none"


class RegressorMetrics(str, Enum):
    """Regressor metrics available."""

    mse = "mse"
    rmse = "rmse"
    mae = "mae"
    r2 = "r2"


class ClassifierMetrics(str, Enum):
    """Classifier metrics available."""

    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    f1 = "f1"
    auc = "auc"


class LogisticRegressionPenalties(str, Enum):
    """Logistic regression penalties available."""

    l1 = "l1"
    l2 = "l2"
    elasicnet = "elasicnet"
    none = "None"


class LogisticRegressionSolvers(str, Enum):
    """Logistic regression solvers available."""

    lbfgs = "lbfgs"
    liblinear = "liblinear"
    newton_cg = "newton-cg"  # '-' converted to '_' for enum syntax
    newton_cholesky = "newton-cholesky"  # '-' converted to '_' for enum syntax
    sag = "sag"
    saga = "saga"


class GradientBoostingClassifierLosses(str, Enum):
    """Gradient boosting classifier losses available."""

    log_loss = "log_loss"
    exponential = "exponential"


class GradientBoostingRegressorLosses(str, Enum):
    """Gradient boosting regressor losses available."""

    squared_error = "squared_error"
    absolute_error = "absolute_error"
    huber = "huber"
    quantile = "quantile"


class NodataHandling(str, Enum):
    """Nodata handling choices."""

    replace = "replace"
    remove = "remove"


class FocalFilterMethod(str, Enum):
    """Focal filter methods."""

    mean = "mean"
    median = "median"


class FocalFilterShape(str, Enum):
    """Shape of the filter window."""

    square = "square"
    circle = "circle"


class MexicanHatFilterDirection(str, Enum):
    """Direction of calculating kernel values."""

    rectangular = "rectangular"
    circular = "circular"


class LocalMoranWeightType(str, Enum):
    """Weight type for Local Moran's I."""

    queen = "queen"
    knn = "knn"


class CorrelationMethod(str, Enum):
    """Correlation methods available."""

    pearson = "pearson"
    kendall = "kendall"
    spearman = "spearman"


class RandomForestClassifierCriterion(str, Enum):
    """Criterion type for Random Forest classifier."""

    gini = "gini"
    entropy = "entropy"
    log_loss = "log_loss"


class RandomForestRegressorCriterion(str, Enum):
    """Criterion type for Random Forest regressor."""

    squared_error = "squared_error"
    absolute_error = "absolute_error"
    friedman_mse = "friedman_mse"
    poisson = "poisson"


class ThresholdCriteria(str, Enum):
    """Threshold criteria for distance to anomaly."""

    lower = "lower"
    higher = "higher"
    in_between = "in_between"
    outside = "outside"


INPUT_FILE_OPTION = Annotated[
    Path,
    typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
]

INPUT_FILES_ARGUMENT = Annotated[
    List[Path],
    typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        help="Paths to input files.",
    ),
]

OUTPUT_FILE_OPTION = Annotated[
    Path,
    typer.Option(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
]

OUTPUT_DIR_OPTION = Annotated[
    Path,
    typer.Option(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
]


def get_enum_values(parameter: Union[Enum, List[Enum]]) -> Union[str, List[str]]:
    """Get values behind enum parameter definition (required for list enums)."""
    if isinstance(parameter, List):
        return [list_item.value for list_item in parameter]
    else:
        return parameter.value


# --- EXPLORATORY ANALYSES ---

# NORMALITY TEST RASTER
@app.command()
def normality_test_raster_cli(input_raster: INPUT_FILE_OPTION, bands: Optional[List[int]] = None):
    """Compute Shapiro-Wilk test for normality on the input raster data."""
    from eis_toolkit.exploratory_analyses.normality_test import normality_test_array

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        data = raster.read()
        typer.echo("Progress: 25%")
        if len(bands) == 0:
            bands = None
        results_dict = normality_test_array(data=data, bands=bands, nodata_value=raster.nodata)

    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo("Normality test (raster) completed")


# NORMALITY TEST VECTOR
@app.command()
def normality_test_vector_cli(input_vector: INPUT_FILE_OPTION, columns: Optional[List[str]] = None):
    """Compute Shapiro-Wilk test for normality on the input vector data."""
    from eis_toolkit.exploratory_analyses.normality_test import normality_test_dataframe

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    results_dict = normality_test_dataframe(data=geodataframe, columns=columns)

    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo("Normality test (vector) completed")


# CHI-SQUARE_TEST
@app.command()
def chi_square_test_cli(
    input_vector: INPUT_FILE_OPTION,
    target_column: str = typer.Option(),
    columns: Optional[List[str]] = None,
):
    """Perform a Chi-square test of independence between a target variable and one or more other variables."""
    from eis_toolkit.exploratory_analyses.chi_square_test import chi_square_test

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)  # Should we drop geometry columns?
    typer.echo("Progress: 25%")

    results_dict = chi_square_test(data=geodataframe, target_column=target_column, columns=columns)

    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo("Chi-square test completed")


# CORRELATION MATRIX
@app.command()
def correlation_matrix_cli(
    input_vector: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    columns: Optional[List[str]] = None,
    correlation_method: Annotated[CorrelationMethod, typer.Option(case_sensitive=False)] = CorrelationMethod.pearson,
    min_periods: Optional[int] = None,
):
    """Compute correlation matrix on the input data."""
    from eis_toolkit.exploratory_analyses.correlation_matrix import correlation_matrix

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    output_df = correlation_matrix(
        data=dataframe, columns=columns, correlation_method=get_enum_values(correlation_method), min_periods=min_periods
    )

    typer.echo("Progress: 75%")

    output_df.to_csv(output_file)
    typer.echo("Progress: 100%")

    typer.echo("Correlation matrix completed")


# COVARIANCE MATRIX
@app.command()
def covariance_matrix_cli(
    input_vector: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    columns: Optional[List[str]] = None,
    min_periods: Optional[int] = None,
    delta_degrees_of_freedom: int = 1,
):
    """Compute covariance matrix on the input data."""
    from eis_toolkit.exploratory_analyses.covariance_matrix import covariance_matrix

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    output_df = covariance_matrix(
        data=dataframe, columns=columns, min_periods=min_periods, delta_degrees_of_freedom=delta_degrees_of_freedom
    )

    typer.echo("Progress: 75%")

    output_df.to_csv(output_file)
    typer.echo("Progress: 100%")

    typer.echo("Covariance matrix completed")


# DBSCAN VECTOR
@app.command()
def dbscan_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    include_coordinates: bool = True,
    columns: Annotated[List[str], typer.Option()] = None,
    max_distance: float = 0.5,
    min_samples: int = 5,
):
    """Perform DBSCAN clustering on the input vector data."""
    from eis_toolkit.exploratory_analyses.dbscan import dbscan_vector

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    output_geodataframe = dbscan_vector(
        data=geodataframe,
        include_coordinates=include_coordinates,
        columns=columns,
        max_distance=max_distance,
        min_samples=min_samples,
    )
    typer.echo("Progress: 75%")

    output_geodataframe.to_file(output_vector, driver="GPKG")
    typer.echo("Progress: 100%")

    typer.echo(f"DBSCAN completed, output vector written to {output_vector}.")


# DBSCAN RASTER
@app.command()
def dbscan_raster_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
    max_distance: float = 0.5,
    min_samples: int = 5,
):
    """Perform DBSCAN clustering on the input raster data."""
    from eis_toolkit.exploratory_analyses.dbscan import dbscan_array
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")
    typer.echo("Progress: 25%")

    output_array = dbscan_array(data=stacked_array, max_distance=max_distance, min_samples=min_samples)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["nodata"] = -9999
    out_profile["count"] = 1

    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(output_array, 1)

    typer.echo("Progress: 100%")
    typer.echo(f"DBSCAN clustering completed, output raster written to {output_raster}.")


# K-MEANS CLUSTERING VECTOR
@app.command()
def k_means_clustering_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    include_coordinates: bool = True,
    columns: Annotated[List[str], typer.Option()] = None,
    number_of_clusters: Optional[int] = None,
    random_state: int = None,
):
    """Perform k-means clustering on the input vector data."""
    from eis_toolkit.exploratory_analyses.k_means_cluster import k_means_clustering_vector

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    output_geodataframe = k_means_clustering_vector(
        data=geodataframe,
        include_coordinates=include_coordinates,
        columns=columns,
        number_of_clusters=number_of_clusters,
        random_state=random_state,
    )
    typer.echo("Progress: 75%")

    output_geodataframe.to_file(output_vector, driver="GPKG")
    typer.echo("Progress: 100%")

    typer.echo(f"K-means clustering completed, output vector written to {output_vector}.")


# K-MEANS CLUSTERING RASTER
@app.command()
def k_means_clustering_raster_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_clusters: Optional[int] = None,
    random_state: int = None,
):
    """Perform k-means clustering on the input raster data."""
    from eis_toolkit.exploratory_analyses.k_means_cluster import k_means_clustering_array
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")
    typer.echo("Progress: 25%")

    output_array = k_means_clustering_array(
        data=stacked_array, number_of_clusters=number_of_clusters, random_state=random_state
    )
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["nodata"] = -9999
    out_profile["count"] = 1

    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(output_array, 1)

    typer.echo("Progress: 100%")
    typer.echo(f"K-means clustering completed, output raster written to {output_raster}.")


# PARALLEL COORDINATES
@app.command()
def parallel_coordinates_cli(
    input_vector: INPUT_FILE_OPTION,
    output_file: Optional[OUTPUT_FILE_OPTION] = None,
    color_column_name: str = typer.Option(),
    plot_title: Optional[str] = None,
    palette_name: Optional[str] = None,
    curved_lines: bool = True,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """Generate a parallel coordinates plot."""
    import matplotlib.pyplot as plt

    from eis_toolkit.exploratory_analyses.parallel_coordinates import plot_parallel_coordinates

    typer.echo("Progress: 10%")
    geodataframe = gpd.read_file(input_vector)
    dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    _ = plot_parallel_coordinates(
        dataframe,
        color_column_name=color_column_name,
        plot_title=plot_title,
        palette_name=palette_name,
        curved_lines=curved_lines,
    )
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    echo_str_end = "."
    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100%")

    typer.echo("Parallel coordinates plot completed" + echo_str_end)


# PCA FOR RASTER DATA
@app.command()
def compute_pca_raster_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_components: int = typer.Option(),
    # NOTE: Omitted scaler type selection here since the parameter might be deleted from PCA func
    nodata_handling: Annotated[NodataHandling, typer.Option(case_sensitive=False)] = NodataHandling.remove,
    # NOTE: Omitted nodata parameter. Should use raster nodata.
):
    """Compute defined number of principal components for raster data."""
    from eis_toolkit.exploratory_analyses.pca import compute_pca
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")
    typer.echo("Progress: 25%")

    pca_array, variance_ratios = compute_pca(
        data=stacked_array, number_of_components=number_of_components, nodata_handling=get_enum_values(nodata_handling)
    )

    # Fill np.nan with nodata before writing data to raster
    pca_array[pca_array == np.nan] = -9999
    out_profile = profiles[0]
    out_profile["nodata"] = -9999

    # Update nr of bands
    out_profile["count"] = number_of_components

    # Create dictionary from the variance ratios array
    variances_ratios_dict = {}
    for i, variance_ratio in enumerate(variance_ratios):
        name = "PC " + str(i) + " explained variance"
        variances_ratios_dict[name] = variance_ratio
    json_str = json.dumps(variances_ratios_dict)

    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(pca_array)

    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo(f"PCA computation (raster) completed, output raster saved to {output_raster}.")


# PCA FOR VECTOR DATA
@app.command()
def compute_pca_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    number_of_components: int = typer.Option(),
    columns: Annotated[List[str], typer.Option()] = None,
    # NOTE: Omitted scaler type selection here since the parameter might be deleted from PCA func
    nodata_handling: Annotated[NodataHandling, typer.Option(case_sensitive=False)] = NodataHandling.remove,
    nodata: float = None,
):
    """Compute defined number of principal components for vector data."""
    from eis_toolkit.exploratory_analyses.pca import compute_pca

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    pca_gdf, variance_ratios = compute_pca(
        data=gdf,
        number_of_components=number_of_components,
        columns=columns,
        nodata_handling=get_enum_values(nodata_handling),
        nodata=nodata,
    )

    # Create dictionary from the variance ratios array
    variances_ratios_dict = {}
    for i, variance_ratio in enumerate(variance_ratios):
        name = "PC " + str(i) + " explained variance"
        variances_ratios_dict[name] = variance_ratio
    json_str = json.dumps(variances_ratios_dict)

    pca_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo(f"PCA computation (vector) completed, output vector saved to {output_vector}.")


# DESCRIPTIVE STATISTICS (RASTER)
@app.command()
def descriptive_statistics_raster_cli(input_file: INPUT_FILE_OPTION):
    """Generate descriptive statistics from raster data."""
    from eis_toolkit.exploratory_analyses.descriptive_statistics import descriptive_statistics_raster

    typer.echo("Progress: 10%")

    with rasterio.open(input_file) as raster:
        typer.echo("Progress: 25%")
        results_dict = descriptive_statistics_raster(raster)
    typer.echo("Progress: 75%")

    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {str(results_dict)}")
    typer.echo("\nDescriptive statistics (raster) completed")


# DESCRIPTIVE STATISTICS (VECTOR)
@app.command()
def descriptive_statistics_vector_cli(input_file: INPUT_FILE_OPTION, column: str = None):
    """Generate descriptive statistics from vector or tabular data."""
    from eis_toolkit.exploratory_analyses.descriptive_statistics import descriptive_statistics_dataframe

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

    typer.echo("Progress: 100%")

    typer.echo(f"Results: {str(results_dict)}")
    typer.echo("\nDescriptive statistics (vector) completed")


# LOCAL MORAN'S I
@app.command()
def local_morans_i_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    column: str = typer.Option(),
    weight_type: Annotated[LocalMoranWeightType, typer.Option(case_sensitive=False)] = LocalMoranWeightType.queen,
    k: int = 4,
    permutations: int = 999,
):
    """Execute Local Moran's I calculation for the data."""
    from eis_toolkit.exploratory_analyses.local_morans_i import local_morans_i

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    out_gdf = local_morans_i(gdf, column, get_enum_values(weight_type), k, permutations)
    typer.echo("Progress: 75%")

    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Local Moran's I completed, output vector saved to {output_vector}.")


# --- RASTER PROCESSING ---


# FOCAL FILTER
@app.command()
def focal_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    method: FocalFilterMethod = FocalFilterMethod.mean,
    size: int = 3,
    shape: Annotated[FocalFilterShape, typer.Option(case_sensitive=False)] = FocalFilterShape.circle,
):
    """Apply a basic focal filter to the input raster."""
    from eis_toolkit.raster_processing.filters.focal import focal_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = focal_filter(raster=raster, method=method, size=size, shape=get_enum_values(shape))
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Focal filter applied, output raster written to {output_raster}.")


# GAUSSIAN FILTER
@app.command()
def gaussian_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    sigma: float = 1.0,
    truncate: float = 4.0,
    size: int = None,
):
    """Apply a gaussian filter to the input raster."""
    from eis_toolkit.raster_processing.filters.focal import gaussian_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = gaussian_filter(raster=raster, sigma=sigma, truncate=truncate, size=size)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Gaussial filter applied, output raster written to {output_raster}.")


# MEXICAN HAT FILTER
@app.command()
def mexican_hat_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    sigma: float = 1.0,
    truncate: float = 4.0,
    size: int = None,
    direction: Annotated[
        MexicanHatFilterDirection, typer.Option(case_sensitive=False)
    ] = MexicanHatFilterDirection.circular,
):
    """Apply a mexican hat filter to the input raster."""
    from eis_toolkit.raster_processing.filters.focal import mexican_hat_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = mexican_hat_filter(
            raster=raster, sigma=sigma, truncate=truncate, size=size, direction=get_enum_values(direction)
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Mexican hat filter applied, output raster written to {output_raster}.")


# LEE ADDITIVE NOISE FILTER
@app.command()
def lee_additive_noise_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    add_noise_var: float = 0.25,
):
    """Apply a Lee filter considering additive noise components in the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import lee_additive_noise_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = lee_additive_noise_filter(raster=raster, size=size, add_noise_var=add_noise_var)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Additive Lee noise filter applied, output raster written to {output_raster}.")


# LEE MULTIPLICATIVE NOISE FILTER
@app.command()
def lee_multiplicative_noise_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    multi_noise_mean: float = 1.0,
    n_looks: int = 1,
):
    """Apply a Lee filter considering multiplicative noise components in the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import lee_multiplicative_noise_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = lee_multiplicative_noise_filter(
            raster=raster, size=size, mult_noise_mean=multi_noise_mean, n_looks=n_looks
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Multiplicative Lee noise filter applied, output raster written to {output_raster}.")


# LEE ADDITIVE MULTIPLICATIVE NOISE FILTER
@app.command()
def lee_additive_multiplicative_noise_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    add_noise_var: float = 0.25,
    add_noise_mean: float = 0,
    multi_noise_mean: float = 1.0,
):
    """Apply a Lee filter considering both additive and multiplicative noise components in the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import lee_additive_multiplicative_noise_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = lee_additive_multiplicative_noise_filter(
            raster=raster,
            size=size,
            add_noise_var=add_noise_var,
            add_noise_mean=add_noise_mean,
            mult_noise_mean=multi_noise_mean,
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Additive multiplicative Lee noise filter applied, output raster written to {output_raster}.")


# LEE ENHANCED FILTER
@app.command()
def lee_enhanced_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    n_looks: int = 1,
    damping_factor: float = 1.0,
):
    """Apply an enhanced Lee filter to the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import lee_enhanced_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = lee_enhanced_filter(
            raster=raster, size=size, n_looks=n_looks, damping_factor=damping_factor
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Enhanced Lee filter applied, output raster written to {output_raster}.")


# GAMMA FILTER
@app.command()
def gamma_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    n_looks: int = 1,
):
    """Apply a Gamma filter to the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import gamma_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = gamma_filter(raster=raster, size=size, n_looks=n_looks)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Gamma filter applied, output raster written to {output_raster}.")


# FROST FILTER
@app.command()
def frost_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    damping_factor: float = 1.0,
):
    """Apply a Frost filter to the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import frost_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = frost_filter(raster=raster, size=size, damping_factor=damping_factor)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Frost filter applied, output raster written to {output_raster}.")


# KUAN FILTER
@app.command()
def kuan_filter_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    size: int = 3,
    n_looks: int = 1,
):
    """Apply a Kuan filter to the input raster."""
    from eis_toolkit.raster_processing.filters.speckle import kuan_filter

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("progress: 25%")
        out_image, out_meta = kuan_filter(raster=raster, size=size, n_looks=n_looks)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Kuan filter applied, output raster written to {output_raster}.")


# CHECK RASTER GRIDS
@app.command()
def check_raster_grids_cli(input_rasters: INPUT_FILES_ARGUMENT, same_extent: bool = False):
    """Check all input rasters for matching gridding and optionally matching bounds."""
    from eis_toolkit.utilities.checks.raster import check_raster_grids

    typer.echo("Progress: 10%")

    raster_profiles = []
    for input_raster in input_rasters:
        with rasterio.open(input_raster) as raster:
            raster_profiles.append(raster.profile)
    typer.echo("Progress: 50%")

    result = check_raster_grids(raster_profiles=raster_profiles, same_extent=same_extent)
    results_dict = {"result": result}
    typer.echo("Progress: 75%")

    json_str = json.dumps(results_dict)
    typer.echo("Progress: 100%")

    typer.echo(f"Results: {json_str}")
    typer.echo("Checking raster grids completed.")


# CLIP RASTER
@app.command()
def clip_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    geometries: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
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


# CREATE CONSTANT RASTER MANUALLY
@app.command()
def create_constant_raster_manually_cli(
    output_raster: OUTPUT_FILE_OPTION,
    constant_value: float = typer.Option(),
    target_epsg: int = typer.Option(),
    extent: Tuple[float, float, float, float] = typer.Option(),
    target_pixel_size: int = typer.Option(),
    nodata_value: float = typer.Option(),
):
    """
    Create constant raster manually by defining CRS, extent and pixel size.

    If the resulting raster height and width are not exact multiples of the pixel size, \
    the output raster extent will differ slightly from the defined extent.
    """
    from eis_toolkit.raster_processing.create_constant_raster import create_constant_raster

    typer.echo("Progress: 10%")

    coord_west, coord_east, coord_south, coord_north = extent
    raster_width = round(abs(coord_east - coord_west) / target_pixel_size)
    raster_height = round(abs(coord_north - coord_south) / target_pixel_size)
    out_image, out_meta = create_constant_raster(
        constant_value=constant_value,
        coord_west=coord_west,
        coord_north=coord_north,
        coord_east=coord_east,
        coord_south=coord_south,
        target_epsg=target_epsg,
        raster_width=raster_width,
        raster_height=raster_height,
        nodata_value=nodata_value,
    )

    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Creating constant raster completed, writing raster to {output_raster}.")


# CREATE CONSTANT RASTER FROM TEMPLATE
@app.command()
def create_constant_raster_from_template_cli(
    template_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    constant_value: float = typer.Option(),
    nodata_value: float = typer.Option(),
):
    """Create constant raster from a template raster."""
    from eis_toolkit.raster_processing.create_constant_raster import create_constant_raster

    typer.echo("Progress: 10%")

    with rasterio.open(template_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = create_constant_raster(
            constant_value=constant_value,
            template_raster=raster,
            nodata_value=nodata_value,
        )

    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Creating constant raster completed, writing raster to {output_raster}.")


# DISTANCE TO ANOMALY
@app.command()
def distance_to_anomaly_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    threshold_criteria: Annotated[ThresholdCriteria, typer.Option(case_sensitive=False)],
    first_threshold_criteria_value: float = typer.Option(),
    second_threshold_criteria_value: float = None,
    max_distance: float = None,
):
    """
    Calculate distance from each raster cell to nearest anomaly cell.

    Uses only the first band of the raster.
    """
    from eis_toolkit.raster_processing.distance_to_anomaly import distance_to_anomaly

    typer.echo("Progress: 10%")

    if second_threshold_criteria_value is not None:
        threshold_criteria_value = (first_threshold_criteria_value, second_threshold_criteria_value)
    else:
        threshold_criteria_value = first_threshold_criteria_value

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = distance_to_anomaly(
            anomaly_raster_profile=raster.profile,
            anomaly_raster_data=raster.read(1),
            threshold_criteria_value=threshold_criteria_value,
            threshold_criteria=get_enum_values(threshold_criteria),
            max_distance=max_distance,
        )

    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Computing distance to anomaly completed, writing raster to {output_raster}.")


# EXTRACT VALUES FROM RASTER
@app.command()
def extract_values_from_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    geometries: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
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
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    target_crs: int = typer.Option(help="crs help"),
    resampling_method: Annotated[ResamplingMethods, typer.Option(case_sensitive=False)] = ResamplingMethods.nearest,
):
    """Reproject the input raster to given CRS."""
    from eis_toolkit.raster_processing.reprojecting import reproject_raster

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reproject_raster(
            raster=raster, target_crs=target_crs, resampling_method=get_enum_values(resampling_method)
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reprojecting completed, writing raster to {output_raster}.")


# RESAMPLE RASTER
@app.command()
def resample_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    resolution: float = typer.Option(),
    resampling_method: Annotated[ResamplingMethods, typer.Option(case_sensitive=False)] = ResamplingMethods.bilinear,
):
    """Resamples raster according to given resolution."""
    from eis_toolkit.raster_processing.resampling import resample

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = resample(
            raster=raster, resolution=resolution, resampling_method=get_enum_values(resampling_method)
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress 100%")

    typer.echo(f"Resampling completed, writing raster to {output_raster}.")


# SNAP RASTER
@app.command()
def snap_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    snap_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
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
    rasters_to_unify: INPUT_FILES_ARGUMENT,
    base_raster: INPUT_FILE_OPTION,
    output_directory: OUTPUT_DIR_OPTION,
    resampling_method: Annotated[ResamplingMethods, typer.Option(case_sensitive=False)] = ResamplingMethods.nearest,
    same_extent: bool = False,
):
    """Unify rasters to match the base raster."""
    from eis_toolkit.raster_processing.unifying import unify_raster_grids

    typer.echo("Progress: 10%")

    with rasterio.open(base_raster) as raster:
        to_unify = [rasterio.open(rstr) for rstr in rasters_to_unify]  # Open all rasters to be unified
        typer.echo("Progress: 25%")

        unified = unify_raster_grids(
            base_raster=raster,
            rasters_to_unify=to_unify,
            resampling_method=get_enum_values(resampling_method),
            same_extent=same_extent,
        )
        [rstr.close() for rstr in to_unify]  # Close all rasters
    typer.echo("Progress: 75%")

    out_rasters_dict = {}
    for i, (out_image, out_meta) in enumerate(unified[1:]):  # Skip writing base raster
        in_raster_name = os.path.splitext(os.path.split(rasters_to_unify[i])[1])[0]
        output_raster_name = f"{in_raster_name}_unified"
        output_raster_path = output_directory.joinpath(output_raster_name + ".tif")
        with rasterio.open(output_raster_path, "w", **out_meta) as dst:
            dst.write(out_image)
        out_rasters_dict[output_raster_name] = str(output_raster_path)
    typer.echo("Progress: 100%")

    json_str = json.dumps(out_rasters_dict)
    typer.echo(f"Output rasters: {json_str}")
    typer.echo(f"Unifying completed, rasters saved to {output_directory}.")


# GET UNIQUE COMBINATIONS
@app.command()
def unique_combinations_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Get combinations of raster values between rasters."""
    from eis_toolkit.raster_processing.unique_combinations import unique_combinations

    typer.echo("Progress: 10%")
    rasters = [rasterio.open(rstr) for rstr in input_rasters]

    typer.echo("Progress: 25%")
    out_image, out_meta = unique_combinations(rasters)
    [rstr.close() for rstr in rasters]
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image, 1)

    typer.echo(f"Writing results to {output_raster}.")
    typer.echo("Getting unique combinations completed.")


# EXTRACT WINDOW
@app.command()
def extract_window_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
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


# SURFACE DERIVATIVES - CLASSIFY ASPECT
@app.command()
def classify_aspect_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    unit: Annotated[AngleUnits, typer.Option(case_sensitive=False)] = AngleUnits.radians,
    num_classes: int = 8,
):
    """Classify an aspect raster data set."""
    from eis_toolkit.raster_processing.derivatives.classification import classify_aspect

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, class_mapping, out_meta = classify_aspect(
            raster=raster, unit=get_enum_values(unit), num_classes=num_classes
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image, 1)
    json_str = json.dumps(class_mapping)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo(f"Classifying aspect completed, writing raster to {output_raster}")


# SURFACE DERIVATIVES
@app.command()
def surface_derivatives_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    first_order_parameters: Annotated[List[SurfaceParameter], typer.Option(case_sensitive=False)],
    second_order_parameters: Annotated[List[SurfaceParameter], typer.Option(case_sensitive=False)],
    scaling_factor: Optional[float] = 1.0,
    slope_tolerance: Optional[float] = 0.0,
    slope_gradient_unit: Annotated[SlopeGradientUnit, typer.Option(case_sensitive=False)] = SlopeGradientUnit.radians,
    slope_direction_unit: Annotated[AngleUnits, typer.Option(case_sensitive=False)] = AngleUnits.radians,
    first_order_method: Annotated[FirstOrderMethod, typer.Option(case_sensitive=False)] = FirstOrderMethod.Horn,
    second_order_method: Annotated[SecondOrderMethod, typer.Option(case_sensitive=False)] = SecondOrderMethod.Young,
):
    """Calculate the first and/or second order surface attributes."""
    from eis_toolkit.raster_processing.derivatives.parameters import first_order, second_order_basic_set

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        if first_order_parameters:
            first_order_results = first_order(
                raster=raster,
                parameters=first_order_parameters,
                scaling_factor=scaling_factor,
                slope_tolerance=slope_tolerance,
                slope_gradient_unit=get_enum_values(slope_gradient_unit),
                slope_direction_unit=get_enum_values(slope_direction_unit),
                method=get_enum_values(first_order_method),
            )

        typer.echo("Progress: 50%")
        if second_order_parameters:
            second_order_results = second_order_basic_set(
                raster=raster,
                parameters=second_order_parameters,
                scaling_factor=scaling_factor,
                slope_tolerance=slope_tolerance,
                method=get_enum_values(second_order_method),
            )
    typer.echo("Progres: 75%")

    if first_order_parameters:
        for parameter, (out_image, out_meta) in first_order_results.items():
            out_raster_name = str(output_raster)[:-4] + "_" + parameter + str(output_raster)[-4:]
            with rasterio.open(out_raster_name, "w", **out_meta) as dest:
                dest.write(out_image, 1)
        typer.echo("Progress: 90%")

    if second_order_parameters:
        for parameter, (out_image, out_meta) in second_order_results.items():
            out_raster_name = str(output_raster)[:-4] + "_" + parameter + str(output_raster)[-4:]
            with rasterio.open(out_raster_name, "w", **out_meta) as dest:
                dest.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Calculating first and/or second order surface attributes completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_manual_breaks_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    breaks: Annotated[List[int], typer.Option()],
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with manual breaks."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_manual_breaks

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_manual_breaks(raster=raster, breaks=breaks, bands=bands)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with manual breaks completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_defined_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    interval_size: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with defined intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_defined_intervals

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_defined_intervals(raster=raster, interval_size=interval_size, bands=bands)
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with defined intervals completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_equal_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_intervals: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with equal intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_equal_intervals

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_equal_intervals(
            raster=raster, number_of_intervals=number_of_intervals, bands=bands
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with equal intervals completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_quantiles_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_quantiles: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with quantiles."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_quantiles

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_quantiles(
            raster=raster, number_of_quantiles=number_of_quantiles, bands=bands
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with quantiles completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_natural_breaks_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_classes: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with natural breaks (Jenks Caspall)."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_natural_breaks

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_natural_breaks(
            raster=raster, number_of_classes=number_of_classes, bands=bands
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with natural breaks completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_geometrical_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_classes: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with geometrical intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_geometrical_intervals

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_geometrical_intervals(
            raster=raster, number_of_classes=number_of_classes, bands=bands
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with geometric intervals completed, writing raster to {output_raster}")


@app.command()
def reclassify_with_standard_deviation_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_intervals: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with standard deviation."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_standard_deviation

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = reclassify_with_standard_deviation(
            raster=raster, number_of_intervals=number_of_intervals, bands=bands
        )
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Reclassification with standard deviation completed, writing raster to {output_raster}")


# --- VECTOR PROCESSING ---


# CALCULATE GEOMETRY
@app.command()
def calculate_geometry_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Calculate the length or area of the given geometries."""
    from eis_toolkit.vector_processing.calculate_geometry import calculate_geometry

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    out_vector = calculate_geometry(geodataframe=geodataframe)
    typer.echo("Progress: 75%")

    out_vector.to_file(output_vector)
    typer.echo("Progress 100%")
    typer.echo(f"Calculate geometry completed, writing vector to {output_vector}")


# EXTRACT SHARED LINES
@app.command()
def extract_shared_lines_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Extract shared lines/borders/edges between polygons."""
    from eis_toolkit.vector_processing.extract_shared_lines import extract_shared_lines

    typer.echo("Progress: 10%")

    polygons = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    out_vector = extract_shared_lines(polygons=polygons)
    typer.echo("Progress: 75%")

    out_vector.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Extracting shared lines completed, writing vector to {out_vector}")


# IDW INTERPOLATION
@app.command()
def idw_interpolation_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION = None,
    target_column: str = typer.Option(),
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
    power: float = 2.0,
):
    """Apply inverse distance weighting (IDW) interpolation to input vector file."""
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.idw_interpolation import idw

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if base_raster is None or base_raster == "":
        if any(bound is None for bound in extent) or pixel_size is None or pixel_size <= 0:
            raise InvalidParameterValueException(
                "Expected positive pixel size and defined extent in absence of base raster. "
                + f"Pixel size: {pixel_size}, extent: {extent}."
            )
        profile = profile_from_extent_and_pixel_size(extent, (pixel_size, pixel_size))
        profile["crs"] = geodataframe.crs
        profile["driver"] = "GTiff"
        profile["dtype"] = "float32"
    else:
        with rasterio.open(base_raster) as raster:
            profile = raster.profile.copy()

    out_image = idw(geodataframe=geodataframe, target_column=target_column, raster_profile=profile, power=power)
    typer.echo("Progress: 75%")

    profile["count"] = 1
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"IDW interpolation completed, writing raster to {output_raster}.")


# KRIGING INTERPOLATION
@app.command()
def kriging_interpolation_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION = None,
    target_column: str = typer.Option(),
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
    variogram_model: Annotated[VariogramModel, typer.Option(case_sensitive=False)] = VariogramModel.linear,
    coordinates_type: Annotated[CoordinatesType, typer.Option(case_sensitive=False)] = CoordinatesType.geographic,
    method: Annotated[KrigingMethod, typer.Option(case_sensitive=False)] = KrigingMethod.ordinary,
):
    """Apply kriging interpolation to input vector file."""
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.kriging_interpolation import kriging

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if base_raster is None or base_raster == "":
        if any(bound is None for bound in extent) or pixel_size is None or pixel_size <= 0:
            raise InvalidParameterValueException(
                "Expected positive pixel size and defined extent in absence of base raster. "
                + f"Pixel size: {pixel_size}, extent: {extent}."
            )
        profile = profile_from_extent_and_pixel_size(extent, (pixel_size, pixel_size))
        profile["crs"] = geodataframe.crs
        profile["driver"] = "GTiff"
        profile["dtype"] = "float32"
    else:
        with rasterio.open(base_raster) as raster:
            profile = raster.profile.copy()

    out_image = kriging(
        geodataframe=geodataframe,
        target_column=target_column,
        raster_profile=profile,
        variogram_model=get_enum_values(variogram_model),
        coordinates_type=get_enum_values(coordinates_type),
        method=get_enum_values(method),
    )
    typer.echo("Progress: 75%")

    profile["count"] = 1
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Kriging interpolation completed, writing raster to {output_raster}.")


# RASTERIZE
@app.command()
def rasterize_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION = None,
    value_column: str = None,
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
    default_value: float = 1.0,
    fill_value: float = 0.0,
    buffer_value: float = None,
    merge_strategy: Annotated[MergeStrategy, typer.Option(case_sensitive=False)] = MergeStrategy.replace,
):
    """
    Rasterize input vector.

    Either base raster or pixel size + extent must be provided.
    """
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if base_raster is None or base_raster == "":
        if any(bound is None for bound in extent) or pixel_size is None or pixel_size <= 0:
            raise InvalidParameterValueException(
                "Expected positive pixel size and defined extent in absence of base raster. "
                + f"Pixel size: {pixel_size}, extent: {extent}."
            )
        profile = profile_from_extent_and_pixel_size(extent, (pixel_size, pixel_size))
        profile["crs"] = geodataframe.crs
        profile["driver"] = "GTiff"
        profile["dtype"] = "float32"
    else:
        with rasterio.open(base_raster) as raster:
            profile = raster.profile.copy()

    out_image = rasterize_vector(
        geodataframe,
        profile,
        value_column,
        default_value,
        fill_value,
        buffer_value,
        get_enum_values(merge_strategy),
    )
    typer.echo("Progress: 75%")

    profile["count"] = 1
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Rasterizing completed, writing raster to {output_raster}.")


# REPROJECT VECTOR
@app.command()
def reproject_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
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
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION = None,
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
    buffer_value: float = None,
    statistic: Annotated[VectorDensityStatistic, typer.Option(case_sensitive=False)] = VectorDensityStatistic.density,
):
    """
    Compute density of geometries within raster.

    Either base raster or pixel size + extent must be provided.
    """
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.vector_density import vector_density

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if base_raster is None or base_raster == "":
        if any(bound is None for bound in extent) or pixel_size is None or pixel_size <= 0:
            raise InvalidParameterValueException(
                "Expected positive pixel size and defined extent in absence of base raster. "
                + f"Pixel size: {pixel_size}, extent: {extent}."
            )
        profile = profile_from_extent_and_pixel_size(extent, (pixel_size, pixel_size))
        profile["crs"] = geodataframe.crs
        profile["driver"] = "GTiff"
        profile["dtype"] = "float32"
    else:
        with rasterio.open(base_raster) as raster:
            profile = raster.profile.copy()

    out_image = vector_density(
        geodataframe=geodataframe,
        raster_profile=profile,
        buffer_value=buffer_value,
        statistic=get_enum_values(statistic),
    )
    typer.echo("Progress: 75%")

    profile["count"] = 1
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Vector density computation completed, writing raster to {output_raster}.")


# DISTANCE COMPUTATION
@app.command()
def distance_computation_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION = None,
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
    max_distance: float = None,
):
    """Calculate distance from raster cell to nearest geometry."""
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.distance_computation import distance_computation

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if base_raster is None or base_raster == "":
        if any(bound is None for bound in extent) or pixel_size is None or pixel_size <= 0:
            raise InvalidParameterValueException(
                "Expected positive pixel size and defined extent in absence of base raster. "
                + f"Pixel size: {pixel_size}, extent: {extent}."
            )
        profile = profile_from_extent_and_pixel_size(extent, (pixel_size, pixel_size))
        profile["crs"] = geodataframe.crs
        profile["driver"] = "GTiff"
        profile["dtype"] = "float32"
    else:
        with rasterio.open(base_raster) as raster:
            profile = raster.profile.copy()

    out_image = distance_computation(geodataframe=geodataframe, raster_profile=profile, max_distance=max_distance)
    profile["count"] = 1
    typer.echo("Progress: 75%")

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Distance computation completed, writing raster to {output_raster}.")


# CBA
@app.command()
def cell_based_association_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    cell_size: int = typer.Option(),
    column: Optional[str] = None,
    subset_target_attribute_values: Optional[List[str]] = None,
    add_name: Optional[str] = None,
    add_buffer: Optional[float] = None,
):
    """Create a CBA matrix."""
    from eis_toolkit.vector_processing.cell_based_association import cell_based_association

    typer.echo("Progress: 10%")

    geodataframe = gpd.read_file(input_vector)
    typer.echo("Progress: 25%")

    if subset_target_attribute_values is not None:
        subset_target_attribute_values = [value.strip() for value in subset_target_attribute_values]

    cell_based_association(
        cell_size=cell_size,
        geodata=[geodataframe],
        output_path=output_raster,
        column=column if column is None else [column],
        subset_target_attribute_values=subset_target_attribute_values
        if subset_target_attribute_values is None
        else [subset_target_attribute_values],
        add_name=add_name if add_name is None else [add_name],
        add_buffer=add_buffer if add_buffer is None else [add_buffer],
    )

    typer.echo("Progress: 100%")

    typer.echo(f"Cell based association completed, writing raster to {output_raster}.")


# --- PREDICTION ---


# LOGISTIC REGRESSION
@app.command()
def logistic_regression_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    validation_method: Annotated[ValidationMethods, typer.Option(case_sensitive=False)] = ValidationMethods.split_once,
    validation_metrics: Annotated[List[ClassifierMetrics], typer.Option(case_sensitive=False)] = [
        ClassifierMetrics.accuracy
    ],
    split_size: float = 0.2,
    cv_folds: int = 5,
    penalty: Annotated[
        LogisticRegressionPenalties, typer.Option(case_sensitive=False)
    ] = LogisticRegressionPenalties.l2,
    max_iter: int = 100,
    solver: Annotated[LogisticRegressionSolvers, typer.Option(case_sensitive=False)] = LogisticRegressionSolvers.lbfgs,
    verbose: int = 0,
    random_state: Optional[int] = None,
):
    """Train and optionally validate a Logistic Regression classifier model using Sklearn."""
    from eis_toolkit.prediction.logistic_regression import logistic_regression_train
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model

    X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    typer.echo("Progress: 30%")

    # Train (and score) the model
    model, metrics_dict = logistic_regression_train(
        X=X,
        y=y,
        validation_method=get_enum_values(validation_method),
        metrics=get_enum_values(validation_metrics),
        split_size=split_size,
        cv_folds=cv_folds,
        penalty=get_enum_values(penalty),
        max_iter=max_iter,
        solver=get_enum_values(solver),
        verbose=verbose,
        random_state=random_state,
    )

    typer.echo("Progress: 80%")

    save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    typer.echo("Progress: 90%")

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo("Logistic regression training completed")


# RANDOM FOREST CLASSIFIER
@app.command()
def random_forest_classifier_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    validation_method: Annotated[ValidationMethods, typer.Option(case_sensitive=False)] = ValidationMethods.split_once,
    validation_metrics: Annotated[List[ClassifierMetrics], typer.Option(case_sensitive=False)] = [
        ClassifierMetrics.accuracy
    ],
    split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    criterion: Annotated[
        RandomForestClassifierCriterion, typer.Option(case_sensitive=False)
    ] = RandomForestClassifierCriterion.gini,
    max_depth: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[int] = None,
):
    """Train and optionally validate a Random Forest classifier model using Sklearn."""
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model
    from eis_toolkit.prediction.random_forests import random_forest_classifier_train

    X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    typer.echo("Progress: 30%")

    # Train (and score) the model
    model, metrics_dict = random_forest_classifier_train(
        X=X,
        y=y,
        validation_method=get_enum_values(validation_method),
        metrics=get_enum_values(validation_metrics),
        split_size=split_size,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        verbose=verbose,
        random_state=random_state,
    )

    typer.echo("Progress: 80%")

    save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    typer.echo("Progress: 90%")

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo("Random forest classifier training completed")


# RANDOM FOREST REGRESSOR
@app.command()
def random_forest_regressor_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    validation_method: Annotated[ValidationMethods, typer.Option(case_sensitive=False)] = ValidationMethods.split_once,
    validation_metrics: Annotated[List[RegressorMetrics], typer.Option(case_sensitive=False)] = [RegressorMetrics.mse],
    split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    criterion: Annotated[
        RandomForestRegressorCriterion, typer.Option(case_sensitive=False)
    ] = RandomForestRegressorCriterion.squared_error,
    max_depth: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[int] = None,
):
    """Train and optionally validate a Random Forest regressor model using Sklearn."""
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model
    from eis_toolkit.prediction.random_forests import random_forest_regressor_train

    X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    typer.echo("Progress: 30%")

    # Train (and score) the model
    model, metrics_dict = random_forest_regressor_train(
        X=X,
        y=y,
        validation_method=get_enum_values(validation_method),
        metrics=get_enum_values(validation_metrics),
        split_size=split_size,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        verbose=verbose,
        random_state=random_state,
    )

    typer.echo("Progress: 80%")

    save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    typer.echo("Progress: 90%")

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo("Random forest regressor training completed")


# GRADIENT BOOSTING CLASSIFIER
@app.command()
def gradient_boosting_classifier_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    validation_method: Annotated[ValidationMethods, typer.Option(case_sensitive=False)] = ValidationMethods.split_once,
    validation_metrics: Annotated[List[ClassifierMetrics], typer.Option(case_sensitive=False)] = [
        ClassifierMetrics.accuracy
    ],
    split_size: float = 0.2,
    cv_folds: int = 5,
    loss: Annotated[
        GradientBoostingClassifierLosses, typer.Option(case_sensitive=False)
    ] = GradientBoostingClassifierLosses.log_loss,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    subsample: float = 1.0,
    verbose: int = 0,
    random_state: Optional[int] = None,
):
    """Train and optionally validate a Gradient boosting classifier model using Sklearn."""
    from eis_toolkit.prediction.gradient_boosting import gradient_boosting_classifier_train
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model

    X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    typer.echo("Progress: 30%")

    # Train (and score) the model
    model, metrics_dict = gradient_boosting_classifier_train(
        X=X,
        y=y,
        validation_method=get_enum_values(validation_method),
        metrics=get_enum_values(validation_metrics),
        split_size=split_size,
        cv_folds=cv_folds,
        loss=get_enum_values(loss),
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        verbose=verbose,
        random_state=random_state,
    )

    typer.echo("Progress: 80%")

    save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    typer.echo("Progress: 90%")

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo("Gradient boosting classifier training completed")


# GRADIENT BOOSTING REGRESSOR
@app.command()
def gradient_boosting_regressor_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    validation_method: Annotated[ValidationMethods, typer.Option(case_sensitive=False)] = ValidationMethods.split_once,
    validation_metrics: Annotated[List[RegressorMetrics], typer.Option(case_sensitive=False)] = [RegressorMetrics.mse],
    split_size: float = 0.2,
    cv_folds: int = 5,
    loss: Annotated[
        GradientBoostingRegressorLosses, typer.Option(case_sensitive=False)
    ] = GradientBoostingRegressorLosses.squared_error,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    subsample: float = 1.0,
    verbose: int = 0,
    random_state: Optional[int] = None,
):
    """Train and optionally validate a Gradient boosting regressor model using Sklearn."""
    from eis_toolkit.prediction.gradient_boosting import gradient_boosting_regressor_train
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model

    X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    typer.echo("Progress: 30%")

    # Train (and score) the model
    model, metrics_dict = gradient_boosting_regressor_train(
        X=X,
        y=y,
        validation_method=get_enum_values(validation_method),
        metrics=get_enum_values(validation_metrics),
        split_size=split_size,
        cv_folds=cv_folds,
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        verbose=verbose,
        random_state=random_state,
    )

    typer.echo("Progress: 80%")

    save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    typer.echo("Progress: 90%")

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100%")
    typer.echo(f"Results: {json_str}")

    typer.echo("Gradient boosting regressor training completed")


# TEST CLASSIFIER ML MODEL
@app.command()
def classifier_test_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    model_file: INPUT_FILE_OPTION,
    output_raster_probability: OUTPUT_FILE_OPTION,
    output_raster_classified: OUTPUT_FILE_OPTION,
    classification_threshold: float = 0.5,
    test_metrics: Annotated[List[ClassifierMetrics], typer.Option(case_sensitive=False)] = [ClassifierMetrics.accuracy],
):
    """Test trained machine learning classifier model by predicting and scoring."""
    from eis_toolkit.evaluation.scoring import score_predictions
    from eis_toolkit.prediction.machine_learning_general import load_model, prepare_data_for_ml, reshape_predictions
    from eis_toolkit.prediction.machine_learning_predict import predict_classifier

    X, y, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters, target_labels)
    typer.echo("Progress: 30%")

    model = load_model(model_file)
    predictions, probabilities = predict_classifier(X, model, classification_threshold, True)
    probabilities_reshaped = reshape_predictions(
        probabilities, reference_profile["height"], reference_profile["width"], nodata_mask
    )
    predictions_reshaped = reshape_predictions(
        predictions, reference_profile["height"], reference_profile["width"], nodata_mask
    )

    metrics_dict = score_predictions(y, predictions, get_enum_values(test_metrics))
    typer.echo("Progress: 80%")

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with rasterio.open(output_raster_probability, "w", **out_profile) as dst:
        dst.write(probabilities_reshaped, 1)
    with rasterio.open(output_raster_classified, "w", **out_profile) as dst:
        dst.write(predictions_reshaped, 1)

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {json_str}")
    typer.echo(
        (
            "Testing classifier model completed, writing rasters to "
            f"{output_raster_probability} and {output_raster_classified}."
        )
    )


# TEST REGRESSOR ML MODEL
@app.command()
def regressor_test_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    model_file: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    test_metrics: Annotated[List[RegressorMetrics], typer.Option(case_sensitive=False)] = [RegressorMetrics.mse],
):
    """Test trained machine learning regressor model by predicting and scoring."""
    from eis_toolkit.evaluation.scoring import score_predictions
    from eis_toolkit.prediction.machine_learning_general import load_model, prepare_data_for_ml, reshape_predictions
    from eis_toolkit.prediction.machine_learning_predict import predict_regressor

    X, y, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters, target_labels)
    typer.echo("Progress: 30%")

    model = load_model(model_file)
    predictions = predict_regressor(X, model)
    predictions_reshaped = reshape_predictions(
        predictions, reference_profile["height"], reference_profile["width"], nodata_mask
    )

    metrics_dict = score_predictions(y, predictions, get_enum_values(test_metrics))
    typer.echo("Progress: 80%")

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(predictions_reshaped, 1)

    json_str = json.dumps(metrics_dict)
    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {json_str}")
    typer.echo(f"Testing regressor model completed, writing raster to {output_raster}.")


# PREDICT WITH TRAINED ML MODEL
@app.command()
def classifier_predict_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    model_file: INPUT_FILE_OPTION,
    output_raster_probability: OUTPUT_FILE_OPTION,
    output_raster_classified: OUTPUT_FILE_OPTION,
    classification_threshold: float = 0.5,
):
    """Predict with a trained machine learning classifier model."""
    from eis_toolkit.prediction.machine_learning_general import load_model, prepare_data_for_ml, reshape_predictions
    from eis_toolkit.prediction.machine_learning_predict import predict_classifier

    X, _, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters)

    typer.echo("Progress: 30%")

    model = load_model(model_file)
    predictions, probabilities = predict_classifier(X, model, classification_threshold, True)
    probabilities_reshaped = reshape_predictions(
        probabilities, reference_profile["height"], reference_profile["width"], nodata_mask
    )
    predictions_reshaped = reshape_predictions(
        predictions, reference_profile["height"], reference_profile["width"], nodata_mask
    )
    typer.echo("Progress: 80%")

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with rasterio.open(output_raster_probability, "w", **out_profile) as dst:
        dst.write(probabilities_reshaped, 1)
    with rasterio.open(output_raster_classified, "w", **out_profile) as dst:
        dst.write(predictions_reshaped, 1)

    typer.echo("Progress: 100%")
    typer.echo(
        (
            "Predicting with classifier model completed, writing rasters to "
            f"{output_raster_probability} and {output_raster_classified}."
        )
    )


# PREDICT WITH TRAINED ML MODEL
@app.command()
def regressor_predict_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    model_file: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Predict with a trained machine learning regressor model."""
    from eis_toolkit.prediction.machine_learning_general import load_model, prepare_data_for_ml, reshape_predictions
    from eis_toolkit.prediction.machine_learning_predict import predict_regressor

    X, _, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters)

    typer.echo("Progress: 30%")

    model = load_model(model_file)
    predictions = predict_regressor(X, model)
    predictions_reshaped = reshape_predictions(
        predictions, reference_profile["height"], reference_profile["width"], nodata_mask
    )

    typer.echo("Progress: 80%")

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(predictions_reshaped, 1)

    typer.echo("Progress: 100%")
    typer.echo(f"Predicting with regressor model completed, writing raster to {output_raster}.")


# FUZZY OVERLAYS

# AND OVERLAY
@app.command()
def and_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'and' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import and_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    data, profiles = read_and_stack_rasters(input_rasters)
    typer.echo("Progress: 25%")

    out_image = and_overlay(data)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999
    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"'And' overlay completed, writing raster to {output_raster}.")


# OR OVERLAY
@app.command()
def or_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'or' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import or_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    data, profiles = read_and_stack_rasters(input_rasters)
    typer.echo("Progress: 25%")

    out_image = or_overlay(data)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999
    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"'Or' overlay completed, writing raster to {output_raster}.")


# PRODUCT OVERLAY
@app.command()
def product_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'product' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import product_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    data, profiles = read_and_stack_rasters(input_rasters)
    typer.echo("Progress: 25%")

    out_image = product_overlay(data)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999
    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"'Product' overlay completed, writing raster to {output_raster}.")


# SUM OVERLAY
@app.command()
def sum_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'sum' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import sum_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    data, profiles = read_and_stack_rasters(input_rasters)
    typer.echo("Progress: 25%")

    out_image = sum_overlay(data)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999
    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"'Sum' overlay completed, writing raster to {output_raster}.")


# GAMMA OVERLAY
@app.command()
def gamma_overlay_cli(input_rasters: INPUT_FILES_ARGUMENT, output_raster: OUTPUT_FILE_OPTION, gamma: float = 0.5):
    """Compute an 'gamma' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import gamma_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    typer.echo("Progress: 10%")

    data, profiles = read_and_stack_rasters(input_rasters)
    typer.echo("Progress: 25%")

    out_image = gamma_overlay(data, gamma)
    typer.echo("Progress: 75%")

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999
    with rasterio.open(output_raster, "w", **out_profile) as dst:
        dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"'Gamma' overlay completed, writing raster to {output_raster}.")


# WOFE
# TODO


# --- TRANSFORMATIONS ---


# CODA - ALR TRANSFORM
@app.command()
def alr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    column: str = None,
    keep_denominator_column: bool = False,
):
    """Perform an additive logratio transformation on the data."""
    from eis_toolkit.transformations.coda.alr import alr_transform

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_df = alr_transform(df=df, column=column, keep_denominator_column=keep_denominator_column)
    typer.echo("Progess 75%")

    out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"ALR transform completed, output saved to {output_vector}")


# CODA - INVERSE ALR TRANSFORM
@app.command()
def inverse_alr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    denominator_column: str = typer.Option(),
    scale: float = 1.0,
):
    """Perform the inverse transformation for a set of ALR transformed data."""
    from eis_toolkit.transformations.coda.alr import inverse_alr

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_df = inverse_alr(df=df, denominator_column=denominator_column, scale=scale)
    typer.echo("Progess 75%")

    out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Inverse ALR transform completed, output saved to {output_vector}")


# CODA - CLR TRANSFORM
@app.command()
def clr_transform_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Perform a centered logratio transformation on the data."""
    from eis_toolkit.transformations.coda.clr import clr_transform

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_df = clr_transform(df=df)
    typer.echo("Progess 75%")

    out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"CLR transform completed, output saved to {output_vector}")


# CODA - INVERSE CLR TRANSFORM
@app.command()
def inverse_clr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    colnames: Annotated[List[str], typer.Option()] = None,
    scale: float = 1.0,
):
    """Perform the inverse transformation for a set of CLR transformed data."""
    from eis_toolkit.transformations.coda.clr import inverse_clr

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_df = inverse_clr(df=df, colnames=colnames, scale=scale)
    typer.echo("Progess 75%")

    out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Inverse CLR transform completed, output saved to {output_vector}")


# CODA - SINGLE ILR TRANSFORM
@app.command()
def single_ilr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    subcomposition_1: Annotated[List[str], typer.Option()],
    subcomposition_2: Annotated[List[str], typer.Option()],
):
    """Perform a single isometric logratio transformation on the provided subcompositions."""
    from eis_toolkit.transformations.coda.ilr import single_ilr_transform

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_series = single_ilr_transform(df=df, subcomposition_1=subcomposition_1, subcomposition_2=subcomposition_2)
    typer.echo("Progess 75%")

    # NOTE: Output of pairwise_logratio might be changed to DF in the future, to automatically do the following
    df["single_ilr"] = out_series
    out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Single ILR transform completed, output saved to {output_vector}")


# CODA - PAIRWISE LOGRATIO TRANSFORM
@app.command()
def pairwise_logratio_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    numerator_column: str = typer.Option(),
    denominator_column: str = typer.Option(),
):
    """Perform a pairwise logratio transformation on the given columns."""
    from eis_toolkit.transformations.coda.pairwise import pairwise_logratio

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_series = pairwise_logratio(df=df, numerator_column=numerator_column, denominator_column=denominator_column)
    typer.echo("Progess 75%")

    # NOTE: Output of pairwise_logratio might be changed to DF in the future, to automatically do the following
    df["pairwise_logratio"] = out_series
    out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Pairwise logratio transform completed, output saved to {output_vector}")


# CODA - SINGLE PLR TRANSFORM
@app.command()
def single_plr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    column: str = typer.Option(),
):
    """Perform a pivot logratio transformation on the selected column."""
    from eis_toolkit.transformations.coda.plr import single_plr_transform

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_series = single_plr_transform(df=df, column=column)
    typer.echo("Progess 75%")

    # NOTE: Output of single_plr_transform might be changed to DF in the future, to automatically do the following
    df["single_plr"] = out_series
    out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"Single PLR transform completed, output saved to {output_vector}")


# CODA - PLR TRANSFORM
@app.command()
def plr_transform_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Perform a pivot logratio transformation on the dataframe, returning the full set of transforms."""
    from eis_toolkit.transformations.coda.plr import plr_transform

    typer.echo("Progress: 10%")

    gdf = gpd.read_file(input_vector)
    geometries = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    typer.echo("Progress: 25%")

    out_df = plr_transform(df=df)
    typer.echo("Progess 75%")

    out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
    out_gdf.to_file(output_vector)
    typer.echo("Progress: 100%")
    typer.echo(f"PLR transform completed, output saved to {output_vector}")


# BINARIZE
@app.command()
def binarize_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
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
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    limit_lower: Optional[float] = None,
    limit_higher: Optional[float] = None,
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
        out_image, out_meta, _ = clip_transform(raster=raster, limits=[(limit_lower, limit_higher)])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Clip transform completed, writing raster to {output_raster}.")


# Z-SCORE NORMALIZATION
@app.command()
def z_score_normalization_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
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
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    min: float = 0.0,
    max: float = 1.0,
):
    """
    Normalize data based on a specified new range.

    Uses the provided new minimum and maximum to transform data into the new interval.
    """
    from eis_toolkit.transformations.linear import min_max_scaling

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta, _ = min_max_scaling(raster=raster, new_range=[(min, max)])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Min-max scaling completed, writing raster to {output_raster}.")


# LOGARITHMIC
@app.command()
def log_transform_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    log_type: Annotated[LogarithmTransforms, typer.Option(case_sensitive=False)] = LogarithmTransforms.log2,
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
        out_image, out_meta, _ = log_transform(raster=raster, log_transform=[get_enum_values(log_type)])
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Logarithm transform completed, writing raster to {output_raster}.")


# SIGMOID
@app.command()
def sigmoid_transform_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    limit_lower: float = 0.0,
    limit_upper: float = 1.0,
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
        out_image, out_meta, _ = sigmoid_transform(
            raster=raster, bounds=[(limit_lower, limit_upper)], slope=[slope], center=center
        )
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Sigmoid transform completed, writing raster to {output_raster}.")


# WINSORIZE
@app.command()
def winsorize_transform_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    percentile_lower: Optional[float] = None,
    percentile_higher: Optional[float] = None,
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
        out_image, out_meta, _ = winsorize(
            raster=raster, percentiles=[(percentile_lower, percentile_higher)], inside=inside
        )
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Winsorize transform completed, writing raster to {output_raster}.")


# ---EVALUATION ---


@app.command()
def summarize_probability_metrics_cli(true_labels: INPUT_FILE_OPTION, probabilities: INPUT_FILE_OPTION):
    """
    Generate a comprehensive report of various evaluation metrics for classification probabilities.

    The output includes ROC AUC, log loss, average precision and Brier score loss.
    """
    from eis_toolkit.evaluation.classification_probability_evaluation import summarize_probability_metrics
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])
    typer.echo("Progress: 25%")

    results_dict = summarize_probability_metrics(y_true=y_true, y_prob=y_prob)

    typer.echo("Progress: 75%")

    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {str(results_dict)}")
    typer.echo("\nGenerating probability metrics summary completed.")


@app.command()
def summarize_label_metrics_binary_cli(true_labels: INPUT_FILE_OPTION, predictions: INPUT_FILE_OPTION):
    """
    Generate a comprehensive report of various evaluation metrics for binary classification results.

    The output includes accuracy, precision, recall, F1 scores and confusion matrix elements
    (true negatives, false positives, false negatives, true positives).
    """
    from eis_toolkit.evaluation.classification_label_evaluation import summarize_label_metrics_binary
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])
    typer.echo("Progress: 25%")

    results_dict = summarize_label_metrics_binary(y_true=y_true, y_pred=y_pred)
    typer.echo("Progress: 75%")

    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {str(results_dict)}")
    typer.echo("\nGenerating prediction label metrics summary completed.")


@app.command()
def plot_roc_curve_cli(
    true_labels: INPUT_FILE_OPTION,
    probabilities: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """
    Plot ROC (receiver operating characteristic) curve.

    ROC curve is a binary classification multi-threshold metric. The ideal performance corner of the plot
    is top-left. AUC of the ROC curve summarizes model performance across different classification thresholds.
    """
    import matplotlib.pyplot as plt

    from eis_toolkit.evaluation.classification_probability_evaluation import plot_roc_curve
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])
    typer.echo("Progress: 25%")

    _ = plot_roc_curve(y_true=y_true, y_prob=y_prob)
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100% \n")

    typer.echo("ROC curve plot completed" + echo_str_end)


@app.command()
def plot_det_curve_cli(
    true_labels: INPUT_FILE_OPTION,
    probabilities: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """
    Plot DET (detection error tradeoff) curve.

    DET curve is a binary classification multi-threshold metric. DET curves are a variation of ROC curves where
    False Negative Rate is plotted on the y-axis instead of True Positive Rate. The ideal performance corner of
    the plot is bottom-left. When comparing the performance of different models, DET curves can be
    slightly easier to assess visually than ROC curves.
    """
    import matplotlib.pyplot as plt

    from eis_toolkit.evaluation.classification_probability_evaluation import plot_det_curve
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])
    typer.echo("Progress: 25%")

    _ = plot_det_curve(y_true=y_true, y_prob=y_prob)
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100% \n")

    typer.echo("DET curve plot completed" + echo_str_end)


@app.command()
def plot_precision_recall_curve_cli(
    true_labels: INPUT_FILE_OPTION,
    probabilities: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """
    Plot precision-recall curve.

    Precision-recall curve is a binary classification multi-threshold metric. Precision-recall curve shows
    the tradeoff between precision and recall for different classification thresholds.
    It can be a useful measure of success when classes are imbalanced.
    """
    import matplotlib.pyplot as plt

    from eis_toolkit.evaluation.classification_probability_evaluation import plot_precision_recall_curve
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])
    typer.echo("Progress: 25%")

    _ = plot_precision_recall_curve(y_true=y_true, y_prob=y_prob)
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100% \n")

    typer.echo("Precision-Recall curve plot completed" + echo_str_end)


@app.command()
def plot_calibration_curve_cli(
    true_labels: INPUT_FILE_OPTION,
    probabilities: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    n_bins: int = 5,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """
    Plot calibration curve (aka realibity diagram).

    Calibration curve has the frequency of the positive labels on the y-axis and the predicted probability on
    the x-axis. Generally, the close the calibration curve is to line x=y, the better the model is calibrated.
    """
    import matplotlib.pyplot as plt

    from eis_toolkit.evaluation.classification_probability_evaluation import plot_calibration_curve
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])
    typer.echo("Progress: 25%")

    _ = plot_calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=n_bins)
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100% \n")

    typer.echo("Calibration curve plot completed" + echo_str_end)


@app.command()
def plot_confusion_matrix_cli(
    true_labels: INPUT_FILE_OPTION,
    predictions: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    show_plot: bool = False,
    save_dpi: Optional[int] = None,
):
    """Plot confusion matrix to visualize classification results."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    from eis_toolkit.evaluation.plot_confusion_matrix import plot_confusion_matrix
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])
    typer.echo("Progress: 25%")

    matrix = confusion_matrix(y_true, y_pred)
    _ = plot_confusion_matrix(confusion_matrix=matrix)
    typer.echo("Progress: 75%")
    if show_plot:
        plt.show()

    if output_file is not None:
        dpi = "figure" if save_dpi is None else save_dpi
        plt.savefig(output_file, dpi=dpi)
        echo_str_end = f", output figure saved to {output_file}."
    typer.echo("Progress: 100% \n")

    typer.echo("Confusion matrix plot completed" + echo_str_end)


@app.command()
def score_predictions_cli(
    true_labels: INPUT_FILE_OPTION,
    predictions: INPUT_FILE_OPTION,
    metrics: Annotated[List[str], typer.Option()],
):
    """Score predictions."""
    from eis_toolkit.evaluation.scoring import score_predictions
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    typer.echo("Progress: 10%")

    (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])
    typer.echo("Progress: 25%")

    outputs = score_predictions(y_true, y_pred, metrics)
    typer.echo("Progress: 100% \n")

    typer.echo(f"Results: {str(outputs)}")
    typer.echo("\nScoring predictions completed.")


# --- UTILITIES ---
@app.command()
def split_raster_bands_cli(input_raster: INPUT_FILE_OPTION, output_dir: OUTPUT_DIR_OPTION):  # CHECK
    """Split multiband raster into singleband rasters."""
    from eis_toolkit.utilities.file_io import get_output_paths_from_common_name
    from eis_toolkit.utilities.raster import split_raster_bands

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        output_singleband_rasters = split_raster_bands(raster)
    typer.echo("Progress: 70%")

    name = os.path.splitext(os.path.basename(input_raster))[0]
    output_paths = get_output_paths_from_common_name(output_singleband_rasters, output_dir, f"{name}_split", ".tif")
    for output_path, (out_image, out_profile) in zip(output_paths, output_singleband_rasters):
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(out_image, 1)
    typer.echo("Progress: 100%")

    typer.echo(f"Splitting raster completed, writing rasters to directory {output_dir}.")


@app.command()
def combine_raster_bands_cli(input_rasters: INPUT_FILES_ARGUMENT, output_raster: OUTPUT_FILE_OPTION):
    """Combine multiple rasters into one multiband raster."""
    from eis_toolkit.utilities.raster import combine_raster_bands

    typer.echo("Progress: 10%")

    rasters = [rasterio.open(raster) for raster in input_rasters]  # Open all rasters to be combined
    typer.echo("Progress: 25%")

    out_image, out_meta = combine_raster_bands(rasters)
    [raster.close() for raster in rasters]
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Combining rasters completed, writing raster to {output_raster}.")


@app.command()
def unify_raster_nodata_cli(
    input_rasters: INPUT_FILES_ARGUMENT, output_dir: OUTPUT_DIR_OPTION, new_nodata: float = -9999  # CHECK
):
    """Unifies nodata for the input rasters."""
    from eis_toolkit.utilities.file_io import get_output_paths_from_inputs
    from eis_toolkit.utilities.nodata import unify_raster_nodata

    typer.echo("Progress: 10%")

    rasters = [rasterio.open(raster) for raster in input_rasters]  # Open all rasters to be unified
    typer.echo("Progress: 25%")

    unified = unify_raster_nodata(rasters, new_nodata)
    [raster.close() for raster in rasters]
    typer.echo("Progress: 70%")

    output_paths = get_output_paths_from_inputs(input_rasters, output_dir, "nodata_unified", ".tif")
    for output_path, (out_image, out_profile) in zip(output_paths, unified):
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Unifying nodata completed, writing rasters to directory {output_dir}.")


@app.command()
def convert_raster_nodata_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    old_nodata: float = None,
    new_nodata: float = -9999,
):
    """Convert existing nodata values with a new nodata value for a raster."""
    from eis_toolkit.utilities.nodata import convert_raster_nodata

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        typer.echo("Progress: 25%")
        out_image, out_meta = convert_raster_nodata(raster, old_nodata, new_nodata)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Converting nodata completed, writing raster to {output_raster}.")


@app.command()
def set_raster_nodata_cli(
    input_raster: INPUT_FILE_OPTION, output_raster: OUTPUT_FILE_OPTION, new_nodata: float = typer.Option()
):
    """Set new nodata value for raster profile."""
    from eis_toolkit.utilities.nodata import set_raster_nodata

    typer.echo("Progress: 10%")

    with rasterio.open(input_raster) as raster:
        out_image = raster.read()
        typer.echo("Progress: 25%")
        out_meta = set_raster_nodata(raster.meta, new_nodata)
    typer.echo("Progress: 70%")

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)
    typer.echo("Progress: 100%")

    typer.echo(f"Setting nodata completed, writing raster to {output_raster}.")


# if __name__ == "__main__":
def cli():
    """CLI app."""
    app()


if __name__ == "__main__":
    app()
