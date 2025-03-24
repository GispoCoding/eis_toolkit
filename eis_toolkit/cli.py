# --- ! ---
# NOTE! Work in progress in the implementation of command-line interface
# Note also, that this CLI is primarily created for other applications to
# utilize EIS Toolkit, such as EIS QGIS Plugin
# --- ! ---

import json
import os
from contextlib import contextmanager
from enum import Enum
from itertools import zip_longest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import typer
from beartype.typing import List, Optional, Sequence, Tuple, Union
from typing_extensions import Annotated

from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan

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


class MLPActivationFunctions(str, Enum):
    """MLP activation functions."""

    relu = "relu"
    linear = "linear"
    sigmoid = "sigmoid"
    tanh = "tanh"


class MLPClassifierLastActivations(str, Enum):
    """MLP classifier last activation functions."""

    sigmoid = "sigmoid"
    softmax = "softmax"


# class MLPRegressorLastActivations(str, Enum):
#     """MLP regressor last activation functions."""

#     linear = "linear"


class MLPOptimizers(str, Enum):
    """MLP optimizers."""

    adam = "adam"
    adagrad = "adagrad"
    rmsprop = "rmsprop"
    sdg = "sdg"


class MLPClassifierLossFunctions(str, Enum):
    """MLP classifier loss functions."""

    binary_crossentropy = "binary_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"


class MLPRegressorLossFunctions(str, Enum):
    """MLP regressor loss functions."""

    mse = "mse"
    mae = "mae"
    hinge = "hinge"
    huber = "huber"


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


class MaskingMode(str, Enum):
    """Masking modes for raster unification."""

    extents = "extents"
    full = "full"
    none = "none"


class KerasClassifierMetrics(str, Enum):
    """Metrics available for Keras classifier models."""

    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    categorical_crossentropy = "categorical_crossentropy"


class KerasRegressorMetrics(str, Enum):
    """Metrics available for Keras regressor models."""

    mse = "mse"
    mae = "mae"


class WeightsType(str, Enum):
    """Weights type for WofE."""

    unique = "unique"
    categorical = "categorical"
    ascending = "ascending"
    descending = "descending"


class ReplaceCondition(str, Enum):
    """Replace conditions for replace with nodata."""

    equal = "equal"
    less_than = "less_than"
    greater_than = "greater_than"
    less_than_or_equal = "less_than_or_equal"
    greater_than_or_equal = "greater_than_or_equal"


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

INPUT_FILES_OPTION = Annotated[
    List[Path],
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


class ProgressLog:  # noqa: D101
    @contextmanager
    @staticmethod
    def reading_input_files():  # noqa: D102
        typer.echo("Opening input files....")
        yield
        typer.echo("✅ Input files read\n")

    @contextmanager
    @staticmethod
    def running_algorithm():  # noqa: D102
        typer.echo("Running algorithm...")
        yield
        typer.echo("✅ Algorithm run succesfully\n")

    @contextmanager
    @staticmethod
    def saving_output_files(savepath: Union[str, Sequence[str]]):  # noqa: D102
        typer.echo("Saving output files...")
        yield
        if isinstance(savepath, Sequence):
            for file in savepath:
                typer.echo(f"✅ Output file(s) saved to {file}\n")
        else:
            typer.echo(f"✅ Output file(s) saved to {savepath}\n")

    @staticmethod
    def finish():  # noqa: D102
        typer.echo("✅ Algorithm execution finished succesfully\n")


class ResultSender:  # noqa: D101
    @staticmethod
    def send_dict_as_json(dictionary: dict):  # noqa: D102
        typer.echo(f"Results: {json.dumps(dictionary)}")


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

    with ProgressLog.reading_input_files():
        with rasterio.open(input_raster) as raster:
            data = raster.read()
            if len(bands) == 0:
                bands = None

    with ProgressLog.running_algorithm():
        results_dict = normality_test_array(data=data, bands=bands, nodata_value=raster.nodata)

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


# NORMALITY TEST VECTOR
@app.command()
def normality_test_vector_cli(input_vector: INPUT_FILE_OPTION, columns: Optional[List[str]] = None):
    """Compute Shapiro-Wilk test for normality on the input vector data."""
    from eis_toolkit.exploratory_analyses.normality_test import normality_test_dataframe

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        results_dict = normality_test_dataframe(data=geodataframe, columns=columns)

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


# CHI-SQUARE_TEST
@app.command()
def chi_square_test_cli(
    input_vector: INPUT_FILE_OPTION,
    target_column: str = typer.Option(),
    columns: Optional[List[str]] = None,
):
    """Perform a Chi-square test of independence between a target variable and one or more other variables."""
    from eis_toolkit.exploratory_analyses.chi_square_test import chi_square_test

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        results_dict = chi_square_test(data=geodataframe, target_column=target_column, columns=columns)

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)
        dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        output_df = correlation_matrix(
            data=dataframe,
            columns=columns,
            correlation_method=get_enum_values(correlation_method),
            min_periods=min_periods,
        )

    with ProgressLog.saving_output_files(output_file):
        output_df.to_csv(output_file)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)
        dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        output_df = covariance_matrix(
            data=dataframe, columns=columns, min_periods=min_periods, delta_degrees_of_freedom=delta_degrees_of_freedom
        )

    with ProgressLog.saving_output_files(output_file):
        output_df.to_csv(output_file)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        output_geodataframe = dbscan_vector(
            data=geodataframe,
            include_coordinates=include_coordinates,
            columns=columns,
            max_distance=max_distance,
            min_samples=min_samples,
        )

    with ProgressLog.saving_output_files(output_vector):
        output_geodataframe.to_file(output_vector, driver="GPKG")

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")

    with ProgressLog.running_algorithm():
        output_array = dbscan_array(data=stacked_array, max_distance=max_distance, min_samples=min_samples)

    out_profile = profiles[0]
    out_profile["nodata"] = -9999
    out_profile["count"] = 1

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(output_array, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        output_geodataframe = k_means_clustering_vector(
            data=geodataframe,
            include_coordinates=include_coordinates,
            columns=columns,
            number_of_clusters=number_of_clusters,
            random_state=random_state,
        )

    with ProgressLog.saving_output_files(output_vector):
        output_geodataframe.to_file(output_vector, driver="GPKG")

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")

    with ProgressLog.running_algorithm():
        output_array = k_means_clustering_array(
            data=stacked_array, number_of_clusters=number_of_clusters, random_state=random_state
        )

    out_profile = profiles[0]
    out_profile["nodata"] = -9999
    out_profile["count"] = 1

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(output_array, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)
        dataframe = pd.DataFrame(geodataframe.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        _ = plot_parallel_coordinates(
            dataframe,
            color_column_name=color_column_name,
            plot_title=plot_title,
            palette_name=palette_name,
            curved_lines=curved_lines,
        )

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


# PCA FOR RASTER DATA
@app.command()
def compute_pca_raster_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_components: Optional[int] = None,
    # NOTE: Omitted scaler type selection here since the parameter might be deleted from PCA func
    nodata_handling: Annotated[NodataHandling, typer.Option(case_sensitive=False)] = NodataHandling.remove,
    # NOTE: Omitted nodata parameter. Should use raster nodata.
):
    """Compute defined number of principal components for raster data."""
    from eis_toolkit.exploratory_analyses.pca import compute_pca
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    with ProgressLog.reading_input_files():
        stacked_array, profiles = read_and_stack_rasters(input_rasters, nodata_handling="convert_to_nan")

    with ProgressLog.running_algorithm():
        transformed_data, principal_components, variances, variance_ratios = compute_pca(
            data=stacked_array,
            number_of_components=number_of_components,
            nodata_handling=get_enum_values(nodata_handling),
        )

    # Fill np.nan with nodata before writing data to raster
    transformed_data[transformed_data == np.nan] = -9999
    out_profile = profiles[0]
    out_profile["nodata"] = -9999

    # Update nr of bands
    out_profile["count"] = len(variances)

    # Create dictionary from the variance ratios array
    # variances_ratios_dict = {}
    # for i, variance_ratio in enumerate(variance_ratios):
    #     name = "PC " + str(i) + " explained variance"
    #     variances_ratios_dict[name] = variance_ratio
    # json_str = json.dumps(variances_ratios_dict)

    out_dict = {
        "principal_components": np.round(principal_components, 4).tolist(),
        "explained_variances": np.round(variances, 4).tolist(),
        "explained_variance_ratios": np.round(variance_ratios, 4).tolist(),
    }

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(transformed_data)

    ResultSender.send_dict_as_json(out_dict)
    ProgressLog.finish()


# PCA FOR VECTOR DATA
@app.command()
def compute_pca_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    number_of_components: Optional[int] = None,
    columns: Annotated[List[str], typer.Option()] = None,
    # NOTE: Omitted scaler type selection here since the parameter might be deleted from PCA func
    nodata_handling: Annotated[NodataHandling, typer.Option(case_sensitive=False)] = NodataHandling.remove,
    nodata: float = None,
):
    """Compute defined number of principal components for vector data."""
    from eis_toolkit.exploratory_analyses.pca import compute_pca

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        transformed_data, principal_components, variances, variance_ratios = compute_pca(
            data=gdf,
            number_of_components=number_of_components,
            columns=columns,
            nodata_handling=get_enum_values(nodata_handling),
            nodata=nodata,
        )

    # Create dictionary from the variance ratios array
    # variances_ratios_dict = {}
    # for i, variance_ratio in enumerate(variance_ratios):
    #     name = "PC " + str(i) + " explained variance"
    #     variances_ratios_dict[name] = variance_ratio
    # json_str = json.dumps(variances_ratios_dict)

    out_dict = {
        "principal_components": np.round(principal_components, 4).tolist(),
        "explained_variances": np.round(variances, 4).tolist(),
        "explained_variance_ratios": np.round(variance_ratios, 4).tolist(),
    }

    with ProgressLog.saving_output_files(output_vector):
        transformed_data.to_file(output_vector)

    ResultSender.send_dict_as_json(out_dict)
    ProgressLog.finish()


# DESCRIPTIVE STATISTICS (RASTER)
@app.command()
def descriptive_statistics_raster_cli(input_raster: INPUT_FILE_OPTION, band: int = 1):
    """Generate descriptive statistics from raster data."""
    from eis_toolkit.exploratory_analyses.descriptive_statistics import descriptive_statistics_raster

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        results_dict = descriptive_statistics_raster(raster, band)
        raster.close()

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


# DESCRIPTIVE STATISTICS (VECTOR)
@app.command()
def descriptive_statistics_vector_cli(input_file: INPUT_FILE_OPTION, column: str = None):
    """Generate descriptive statistics from vector or tabular data."""
    from eis_toolkit.exploratory_analyses.descriptive_statistics import descriptive_statistics_dataframe

    # TODO modify input file detection
    try:
        with ProgressLog.reading_input_files():
            gdf = gpd.read_file(input_file)
        with ProgressLog.running_algorithm():
            results_dict = descriptive_statistics_dataframe(gdf, column)
    except:  # noqa: E722
        try:
            with ProgressLog.reading_input_files():
                df = pd.read_csv(input_file)
            with ProgressLog.running_algorithm():
                results_dict = descriptive_statistics_dataframe(df, column)
        except:  # noqa: E722
            raise Exception("Could not read input file as geodataframe")

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        out_gdf = local_morans_i(gdf, column, get_enum_values(weight_type), k, permutations)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


# FEATURE IMPORTANCE
@app.command()
def feature_importance_cli(
    model_file: INPUT_FILE_OPTION,
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
):
    """Evaluate the feature importance of a sklearn classifier or regressor."""
    from eis_toolkit.exploratory_analyses.feature_importance import evaluate_feature_importance
    from eis_toolkit.prediction.machine_learning_general import load_model, prepare_data_for_ml

    with ProgressLog.reading_input_files():
        model = load_model(model_file)
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)
        feature_names = [raster.name for raster in input_rasters]

    with ProgressLog.running_algorithm():
        feature_importance, _ = evaluate_feature_importance(model, X, y, feature_names, n_repeats, random_state)

    results = dict(zip(feature_importance["Feature"], feature_importance["Importance"]))

    ResultSender.send_dict_as_json(results)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = focal_filter(raster=raster, method=method, size=size, shape=get_enum_values(shape))
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = gaussian_filter(raster=raster, sigma=sigma, truncate=truncate, size=size)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = mexican_hat_filter(
            raster=raster, sigma=sigma, truncate=truncate, size=size, direction=get_enum_values(direction)
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = lee_additive_noise_filter(raster=raster, size=size, add_noise_var=add_noise_var)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = lee_multiplicative_noise_filter(
            raster=raster, size=size, mult_noise_mean=multi_noise_mean, n_looks=n_looks
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = lee_additive_multiplicative_noise_filter(
            raster=raster,
            size=size,
            add_noise_var=add_noise_var,
            add_noise_mean=add_noise_mean,
            mult_noise_mean=multi_noise_mean,
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = lee_enhanced_filter(
            raster=raster, size=size, n_looks=n_looks, damping_factor=damping_factor
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = gamma_filter(raster=raster, size=size, n_looks=n_looks)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = frost_filter(raster=raster, size=size, damping_factor=damping_factor)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = kuan_filter(raster=raster, size=size, n_looks=n_looks)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


# CHECK RASTER GRIDS
@app.command()
def check_raster_grids_cli(input_rasters: INPUT_FILES_ARGUMENT, same_extent: bool = False):
    """Check all input rasters for matching gridding and optionally matching bounds."""
    from eis_toolkit.utilities.checks.raster import check_raster_grids

    with ProgressLog.reading_input_files():
        raster_profiles = []
        for input_raster in input_rasters:
            with rasterio.open(input_raster) as raster:
                raster_profiles.append(raster.profile)

    with ProgressLog.running_algorithm():
        result = check_raster_grids(raster_profiles=raster_profiles, same_extent=same_extent)

    results_dict = {"result": result}

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


# CLIP RASTER
@app.command()
def clip_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    geometries: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Clip the input raster with geometries in a geodataframe."""
    from eis_toolkit.raster_processing.clipping import clip_raster

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(geometries)
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = clip_raster(
            raster=raster,
            geodataframe=geodataframe,
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(template_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = create_constant_raster(
            constant_value=constant_value,
            template_raster=raster,
            nodata_value=nodata_value,
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


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

    if second_threshold_criteria_value is not None:
        threshold_criteria_value = (first_threshold_criteria_value, second_threshold_criteria_value)
    else:
        threshold_criteria_value = first_threshold_criteria_value

    with ProgressLog.reading_input_files():
        with rasterio.open(input_raster) as raster:
            raster_array = raster.read(1)
            profile = raster.profile.copy()

    # Create nodata mask
    mask = (raster_array == profile["nodata"]) | np.isnan(raster_array)

    with ProgressLog.running_algorithm():
        out_image, out_profile = distance_to_anomaly(
            anomaly_raster_profile=profile,
            anomaly_raster_data=raster_array,
            threshold_criteria_value=threshold_criteria_value,
            threshold_criteria=get_enum_values(threshold_criteria),
            max_distance=max_distance,
        )

    # Apply nodata mask after processing
    out_image[mask] = out_profile["nodata"]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


# PROXIMITY TO ANOMALY
@app.command()
def proximity_to_anomaly_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    threshold_criteria: Annotated[ThresholdCriteria, typer.Option(case_sensitive=False)],
    first_threshold_criteria_value: float = typer.Option(),
    second_threshold_criteria_value: float = None,
    max_distance: float = typer.Option(),
    max_distance_value: float = 0.0,
    anomaly_value: float = 1.0,
):
    """
    Calculate proximity from each raster cell to nearest anomaly cell.

    Uses only the first band of the raster.
    """
    from eis_toolkit.raster_processing.proximity_to_anomaly import proximity_to_anomaly

    if second_threshold_criteria_value is not None:
        threshold_criteria_value = (first_threshold_criteria_value, second_threshold_criteria_value)
    else:
        threshold_criteria_value = first_threshold_criteria_value

    with ProgressLog.reading_input_files():
        with rasterio.open(input_raster) as raster:
            raster_array = raster.read(1)
            profile = raster.profile.copy()

    # Create nodata mask
    mask = (raster_array == profile["nodata"]) | np.isnan(raster_array)

    with ProgressLog.running_algorithm():
        out_image, out_profile = proximity_to_anomaly(
            anomaly_raster_profile=profile,
            anomaly_raster_data=raster_array,
            threshold_criteria_value=threshold_criteria_value,
            threshold_criteria=get_enum_values(threshold_criteria),
            max_distance=max_distance,
            scaling_range=(anomaly_value, max_distance_value),
        )

    # Apply nodata mask
    out_image[mask] = out_profile["nodata"]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dest:
            dest.write(out_image, 1)

    ProgressLog.finish()


# EXTRACT VALUES FROM RASTER
@app.command()
def extract_values_from_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    geometries: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
):
    """Extract raster values using point data to a DataFrame."""
    from eis_toolkit.raster_processing.extract_values_from_raster import extract_values_from_raster

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(geometries)
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        df = extract_values_from_raster(raster_list=[raster], geodataframe=geodataframe)
        raster.close()

    with ProgressLog.saving_output_files(output_vector):
        df.to_csv(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reproject_raster(
            raster=raster, target_crs=target_crs, resampling_method=get_enum_values(resampling_method)
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = resample(
            raster=raster, resolution=resolution, resampling_method=get_enum_values(resampling_method)
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


# SNAP RASTER
@app.command()
def snap_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    snap_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Snaps/aligns input raster to the given snap raster."""
    from eis_toolkit.raster_processing.snapping import snap_with_raster

    with ProgressLog.reading_input_files():
        src = rasterio.open(input_raster)
        snap_src = rasterio.open(snap_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = snap_with_raster(src, snap_src)
        src.close()
        snap_raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


# UNIFY RASTERS
@app.command()
def unify_rasters_cli(
    rasters_to_unify: INPUT_FILES_ARGUMENT,
    base_raster: INPUT_FILE_OPTION,
    output_directory: OUTPUT_DIR_OPTION,
    resampling_method: Annotated[ResamplingMethods, typer.Option(case_sensitive=False)] = ResamplingMethods.nearest,
    masking: Annotated[MaskingMode, typer.Option(case_sensitive=False)] = MaskingMode.extents,
):
    """Unify rasters to match the base raster."""
    from eis_toolkit.raster_processing.unifying import unify_raster_grids

    with ProgressLog.reading_input_files():
        raster = rasterio.open(base_raster)
        to_unify = [rasterio.open(rstr) for rstr in rasters_to_unify]  # Open all rasters to be unified

    with ProgressLog.running_algorithm():
        masking_param = get_enum_values(masking)
        unified = unify_raster_grids(
            base_raster=raster,
            rasters_to_unify=to_unify,
            resampling_method=get_enum_values(resampling_method),
            masking=None if masking_param == "none" else masking_param,
        )
        # Close all rasters
        raster.close()
        [rstr.close() for rstr in to_unify]

    with ProgressLog.saving_output_files(output_directory):
        out_rasters_dict = {}
        for i, (out_image, out_meta) in enumerate(unified[1:]):  # Skip writing base raster
            in_raster_name = os.path.splitext(os.path.split(rasters_to_unify[i])[1])[0]
            output_raster_name = f"{in_raster_name}_unified"
            output_raster_path = output_directory.joinpath(output_raster_name + ".tif")
            with rasterio.open(output_raster_path, "w", **out_meta) as dst:
                dst.write(out_image)
            out_rasters_dict[output_raster_name] = str(output_raster_path)

    ResultSender.send_dict_as_json(out_rasters_dict)
    ProgressLog.finish()


# GET UNIQUE COMBINATIONS
@app.command()
def unique_combinations_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Get combinations of raster values between rasters."""
    from eis_toolkit.raster_processing.unique_combinations import unique_combinations

    with ProgressLog.reading_input_files():
        rasters = [rasterio.open(rstr) for rstr in input_rasters]

    with ProgressLog.running_algorithm():
        out_image, out_meta = unique_combinations(rasters)
        [rstr.close() for rstr in rasters]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = extract_window(raster, center_coords, height, width)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, class_mapping, out_meta = classify_aspect(
            raster=raster, unit=get_enum_values(unit), num_classes=num_classes
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image, 1)

    ResultSender.send_dict_as_json(class_mapping)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
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
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        if first_order_parameters:
            for parameter, (out_image, out_meta) in first_order_results.items():
                out_raster_name = str(output_raster)[:-4] + "_" + parameter + str(output_raster)[-4:]
                with rasterio.open(out_raster_name, "w", **out_meta) as dest:
                    dest.write(out_image, 1)

        if second_order_parameters:
            for parameter, (out_image, out_meta) in second_order_results.items():
                out_raster_name = str(output_raster)[:-4] + "_" + parameter + str(output_raster)[-4:]
                with rasterio.open(out_raster_name, "w", **out_meta) as dest:
                    dest.write(out_image, 1)

    ProgressLog.finish()


@app.command()
def mask_raster_cli(
    input_raster: INPUT_FILE_OPTION,
    base_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Mask input raster using the nodata locations from base raster."""
    from eis_toolkit.raster_processing.masking import mask_raster

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)
        base_rstr = rasterio.open(base_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = mask_raster(raster=raster, base_raster=base_rstr)
        raster.close()
        base_rstr.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_manual_breaks_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    breaks: Annotated[List[int], typer.Option()],
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with manual breaks."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_manual_breaks

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_manual_breaks(raster=raster, breaks=breaks, bands=bands)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_defined_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    interval_size: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with defined intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_defined_intervals

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_defined_intervals(raster=raster, interval_size=interval_size, bands=bands)
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_equal_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_intervals: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with equal intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_equal_intervals

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_equal_intervals(
            raster=raster, number_of_intervals=number_of_intervals, bands=bands
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_quantiles_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_quantiles: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with quantiles."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_quantiles

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_quantiles(
            raster=raster, number_of_quantiles=number_of_quantiles, bands=bands
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_natural_breaks_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_classes: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with natural breaks (Jenks Caspall)."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_natural_breaks

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_natural_breaks(
            raster=raster, number_of_classes=number_of_classes, bands=bands
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_geometrical_intervals_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_classes: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with geometrical intervals."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_geometrical_intervals

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_geometrical_intervals(
            raster=raster, number_of_classes=number_of_classes, bands=bands
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


@app.command()
def reclassify_with_standard_deviation_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    number_of_intervals: int = typer.Option(),
    bands: Annotated[List[int], typer.Option()] = None,
):
    """Classify raster with standard deviation."""
    from eis_toolkit.raster_processing.reclassify import reclassify_with_standard_deviation

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = reclassify_with_standard_deviation(
            raster=raster, number_of_intervals=number_of_intervals, bands=bands
        )
        raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    ProgressLog.finish()


# --- VECTOR PROCESSING ---


# CALCULATE GEOMETRY
@app.command()
def calculate_geometry_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Calculate the length or area of the given geometries."""
    from eis_toolkit.vector_processing.calculate_geometry import calculate_geometry

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        out_vector = calculate_geometry(geodataframe=geodataframe)

    with ProgressLog.saving_output_files(output_vector):
        out_vector.to_file(output_vector)

    ProgressLog.finish()


# EXTRACT SHARED LINES
@app.command()
def extract_shared_lines_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Extract shared lines/borders/edges between polygons."""
    from eis_toolkit.vector_processing.extract_shared_lines import extract_shared_lines

    with ProgressLog.reading_input_files():
        polygons = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        out_vector = extract_shared_lines(polygons=polygons)

    with ProgressLog.saving_output_files(output_vector):
        out_vector.to_file(output_vector)

    ProgressLog.finish()


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
    search_radius: Optional[float] = None,
):
    """Apply inverse distance weighting (IDW) interpolation to input vector file."""
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.idw_interpolation import idw

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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

    with ProgressLog.running_algorithm():
        out_image = idw(
            geodataframe=geodataframe,
            target_column=target_column,
            raster_profile=profile,
            power=power,
            search_radius=search_radius,
        )

    profile["count"] = 1
    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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

    with ProgressLog.running_algorithm():
        out_image = kriging(
            geodataframe=geodataframe,
            target_column=target_column,
            raster_profile=profile,
            variogram_model=get_enum_values(variogram_model),
            coordinates_type=get_enum_values(coordinates_type),
            method=get_enum_values(method),
        )

    profile["count"] = 1
    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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

    with ProgressLog.running_algorithm():
        out_image = rasterize_vector(
            geodataframe,
            profile,
            value_column,
            default_value,
            fill_value,
            buffer_value,
            get_enum_values(merge_strategy),
        )

    profile["count"] = 1
    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# REPROJECT VECTOR
@app.command()
def reproject_vector_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    target_crs: int = typer.Option(help="crs help"),
):
    """Reproject the input vector to given CRS."""
    from eis_toolkit.vector_processing.reproject_vector import reproject_vector

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

    with ProgressLog.running_algorithm():
        reprojected_geodataframe = reproject_vector(geodataframe=geodataframe, target_crs=target_crs)

    with ProgressLog.saving_output_files(output_vector):
        reprojected_geodataframe.to_file(output_vector, driver="GeoJSON")

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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

    with ProgressLog.running_algorithm():
        out_image = vector_density(
            geodataframe=geodataframe,
            raster_profile=profile,
            buffer_value=buffer_value,
            statistic=get_enum_values(statistic),
        )

    profile["count"] = 1
    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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
            mask = None
        else:
            with rasterio.open(base_raster) as raster:
                profile = raster.profile.copy()
                raster_array = raster.read(1)
                mask = (raster_array == profile["nodata"]) | np.isnan(raster_array)

    with ProgressLog.running_algorithm():
        out_image, out_profile = distance_computation(
            geodataframe=geodataframe, raster_profile=profile, max_distance=max_distance
        )

    # Apply nodata mask
    if mask is not None:
        out_image[mask] = out_profile["nodata"]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

        if subset_target_attribute_values is not None:
            subset_target_attribute_values = [value.strip() for value in subset_target_attribute_values]

    with ProgressLog.running_algorithm():
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

    ProgressLog.saving_output_files(output_raster)
    ProgressLog.finish()


# PROXIMITY COMPUTATION
@app.command()
def proximity_computation_cli(
    input_vector: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    max_distance: float = typer.Option(),
    max_distance_value: float = 0.0,
    geometries_value: float = 1.0,
    base_raster: INPUT_FILE_OPTION = None,
    pixel_size: float = None,
    extent: Tuple[float, float, float, float] = (None, None, None, None),
):
    """Calculate proximity from raster cell to nearest geometry."""
    from eis_toolkit.exceptions import InvalidParameterValueException
    from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
    from eis_toolkit.vector_processing.proximity_computation import proximity_computation

    with ProgressLog.reading_input_files():
        geodataframe = gpd.read_file(input_vector)

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
            mask = None
        else:
            with rasterio.open(base_raster) as raster:
                profile = raster.profile.copy()
                raster_array = raster.read(1)
                mask = (raster_array == profile["nodata"]) | np.isnan(raster_array)

    with ProgressLog.running_algorithm():
        out_image, out_profile = proximity_computation(
            geodataframe=geodataframe,
            raster_profile=profile,
            maximum_distance=max_distance,
            scale_range=(geometries_value, max_distance_value),
        )

    # Apply nodata mask
    if mask is not None:
        out_image[mask] = out_profile["nodata"]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
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

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)  # NOTE: Check if .joblib needs to be added to save path

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


# MLP CLASSIFIER
@app.command()
def mlp_classifier_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    neurons: Annotated[List[int], typer.Option()],
    activation: Annotated[MLPActivationFunctions, typer.Option(case_sensitive=False)] = MLPActivationFunctions.relu,
    output_neurons: int = 1,
    last_activation: Annotated[
        MLPClassifierLastActivations, typer.Option(case_sensitive=False)
    ] = MLPClassifierLastActivations.sigmoid,
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Annotated[MLPOptimizers, typer.Option(case_sensitive=False)] = MLPOptimizers.adam,
    learning_rate: float = 0.001,
    loss_function: Annotated[
        MLPClassifierLossFunctions, typer.Option(case_sensitive=False)
    ] = MLPClassifierLossFunctions.binary_crossentropy,
    dropout_rate: Optional[float] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    validation_metrics: Annotated[List[KerasClassifierMetrics], typer.Option(case_sensitive=False)] = [
        KerasClassifierMetrics.accuracy
    ],
    validation_split: float = 0.2,
    random_state: Optional[int] = None,
):
    """Train and validate an MLP classifier model using Keras."""
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model
    from eis_toolkit.prediction.mlp import train_MLP_classifier

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
        # Train (and score) the model
        model, metrics_dict = train_MLP_classifier(
            X=X,
            y=y,
            neurons=neurons,
            activation=get_enum_values(activation),
            output_neurons=output_neurons,
            last_activation=get_enum_values(last_activation),
            epochs=epochs,
            batch_size=batch_size,
            optimizer=get_enum_values(optimizer),
            learning_rate=learning_rate,
            loss_function=get_enum_values(loss_function),
            dropout_rate=dropout_rate,
            early_stopping=early_stopping,
            es_patience=es_patience,
            metrics=get_enum_values(validation_metrics),
            validation_split=validation_split,
            random_state=random_state,
        )

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


# MLP REGRESSOR
@app.command()
def mlp_regressor_train_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    target_labels: INPUT_FILE_OPTION,
    output_file: OUTPUT_FILE_OPTION,
    neurons: Annotated[List[int], typer.Option()],
    activation: Annotated[MLPActivationFunctions, typer.Option(case_sensitive=False)] = MLPActivationFunctions.relu,
    output_neurons: int = 1,
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Annotated[MLPOptimizers, typer.Option(case_sensitive=False)] = MLPOptimizers.adam,
    learning_rate: float = 0.001,
    loss_function: Annotated[
        MLPRegressorLossFunctions, typer.Option(case_sensitive=False)
    ] = MLPRegressorLossFunctions.mse,
    dropout_rate: Optional[float] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    validation_metrics: Annotated[List[KerasRegressorMetrics], typer.Option(case_sensitive=False)] = [
        KerasRegressorMetrics.mse
    ],
    validation_split: float = 0.2,
    random_state: Optional[int] = None,
):
    """Train and validate an MLP regressor model using Keras."""
    from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml, save_model
    from eis_toolkit.prediction.mlp import train_MLP_regressor

    with ProgressLog.reading_input_files():
        X, y, _, _ = prepare_data_for_ml(input_rasters, target_labels)

    with ProgressLog.running_algorithm():
        # Train (and score) the model
        model, metrics_dict = train_MLP_regressor(
            X=X,
            y=y,
            neurons=neurons,
            activation=get_enum_values(activation),
            output_neurons=output_neurons,
            last_activation="linear",
            epochs=epochs,
            batch_size=batch_size,
            optimizer=get_enum_values(optimizer),
            learning_rate=learning_rate,
            loss_function=get_enum_values(loss_function),
            dropout_rate=dropout_rate,
            early_stopping=early_stopping,
            es_patience=es_patience,
            metrics=get_enum_values(validation_metrics),
            validation_split=validation_split,
            random_state=random_state,
        )

    with ProgressLog.saving_output_files(output_file):
        save_model(model, output_file)

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters, target_labels)
        model = load_model(model_file)

    with ProgressLog.running_algorithm():
        predictions, probabilities = predict_classifier(X, model, classification_threshold, True)
        probabilities_reshaped = reshape_predictions(
            probabilities, reference_profile["height"], reference_profile["width"], nodata_mask
        )
        predictions_reshaped = reshape_predictions(
            predictions, reference_profile["height"], reference_profile["width"], nodata_mask
        )

        metrics_dict = score_predictions(y, predictions, get_enum_values(test_metrics), decimals=3)

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with ProgressLog.saving_output_files([output_raster_probability, output_raster_classified]):
        with rasterio.open(output_raster_probability, "w", **out_profile) as dst:
            dst.write(probabilities_reshaped, 1)
        with rasterio.open(output_raster_classified, "w", **out_profile) as dst:
            dst.write(predictions_reshaped, 1)

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, y, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters, target_labels)
        model = load_model(model_file)

    with ProgressLog.running_algorithm():
        predictions = predict_regressor(X, model)
        predictions_reshaped = reshape_predictions(
            predictions, reference_profile["height"], reference_profile["width"], nodata_mask
        )

        metrics_dict = score_predictions(y, predictions, get_enum_values(test_metrics), decimals=3)

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(predictions_reshaped, 1)

    ResultSender.send_dict_as_json(metrics_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, _, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters)
        model = load_model(model_file)

    with ProgressLog.running_algorithm():
        predictions, probabilities = predict_classifier(X, model, classification_threshold, True)
        probabilities_reshaped = reshape_predictions(
            probabilities, reference_profile["height"], reference_profile["width"], nodata_mask
        )
        predictions_reshaped = reshape_predictions(
            predictions, reference_profile["height"], reference_profile["width"], nodata_mask
        )

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with ProgressLog.saving_output_files([output_raster_probability, output_raster_classified]):
        with rasterio.open(output_raster_probability, "w", **out_profile) as dst:
            dst.write(probabilities_reshaped, 1)
        with rasterio.open(output_raster_classified, "w", **out_profile) as dst:
            dst.write(predictions_reshaped, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        X, _, reference_profile, nodata_mask = prepare_data_for_ml(input_rasters)
        model = load_model(model_file)

    with ProgressLog.running_algorithm():
        predictions = predict_regressor(X, model)
        predictions_reshaped = reshape_predictions(
            predictions, reference_profile["height"], reference_profile["width"], nodata_mask
        )

    out_profile = reference_profile.copy()
    out_profile.update({"count": 1, "dtype": np.float32})

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(predictions_reshaped, 1)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        data, profiles = read_and_stack_rasters(input_rasters)

    with ProgressLog.running_algorithm():
        out_image = and_overlay(data)

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# OR OVERLAY
@app.command()
def or_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'or' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import or_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    with ProgressLog.reading_input_files():
        data, profiles = read_and_stack_rasters(input_rasters)

    with ProgressLog.running_algorithm():
        out_image = or_overlay(data)

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# PRODUCT OVERLAY
@app.command()
def product_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'product' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import product_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    with ProgressLog.reading_input_files():
        data, profiles = read_and_stack_rasters(input_rasters)

    with ProgressLog.running_algorithm():
        out_image = product_overlay(data)

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# SUM OVERLAY
@app.command()
def sum_overlay_cli(
    input_rasters: INPUT_FILES_ARGUMENT,
    output_raster: OUTPUT_FILE_OPTION,
):
    """Compute an 'sum' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import sum_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    with ProgressLog.reading_input_files():
        data, profiles = read_and_stack_rasters(input_rasters)

    with ProgressLog.running_algorithm():
        out_image = sum_overlay(data)

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# GAMMA OVERLAY
@app.command()
def gamma_overlay_cli(input_rasters: INPUT_FILES_ARGUMENT, output_raster: OUTPUT_FILE_OPTION, gamma: float = 0.5):
    """Compute an 'gamma' overlay operation with fuzzy logic."""
    from eis_toolkit.prediction.fuzzy_overlay import gamma_overlay
    from eis_toolkit.utilities.file_io import read_and_stack_rasters

    with ProgressLog.reading_input_files():
        data, profiles = read_and_stack_rasters(input_rasters)

    with ProgressLog.running_algorithm():
        out_image = gamma_overlay(data, gamma)

    out_profile = profiles[0]
    out_profile["count"] = 1
    out_profile["nodata"] = -9999

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_profile) as dst:
            dst.write(out_image, 1)

    ProgressLog.finish()


# WOFE
@app.command()
def weights_of_evidence_calculate_weights_cli(
    evidential_raster: INPUT_FILE_OPTION,
    deposits: INPUT_FILE_OPTION,
    output_results_table: OUTPUT_FILE_OPTION,
    output_raster_dir: OUTPUT_DIR_OPTION,
    raster_nodata: Optional[float] = None,
    weights_type: Annotated[WeightsType, typer.Option(case_sensitive=False)] = WeightsType.unique,
    studentized_contrast_threshold: float = 1,
    arrays_to_generate: Annotated[Optional[List[str]], typer.Option()] = None,
):
    """
    Calculate weights of spatial associations.

    Save path for resulting CSV is set using --output-results-table parameter. Output rasters are saved to directory
    set with --output-raster-dir parameter.

    Parameter --studentized-contrast-threshold is used with 'categorical', 'ascending' and 'descending' weight types.

    Parameter --arrays-to-generate controls which columns in the weights dataframe are returned as arrays. All column
    names in the produced weights_df are valid choices. The available columns for "unique" weights_type are "Class",
    "Pixel count", "Deposit count", "W+", "S_W+", "W-", "S_W-", "Contrast", "S_Contrast", and "Studentized contrast".
    For other weights types, additional available column names are "Generalized class", "Generalized W+", and
    "Generalized S_W+". Defaults to ["Class", "W+", "S_W+] for "unique" weights_type and ["Class", "W+", "S_W+",
    "Generalized W+", "Generalized S_W+"] for the cumulative weight types.
    """
    from eis_toolkit.prediction.weights_of_evidence import weights_of_evidence_calculate_weights

    with ProgressLog.reading_input_files():
        evidential_raster = rasterio.open(evidential_raster)

        if deposits.suffix in (".tif", ".tiff", ".asc", ".img", ".vrt", ".grd"):
            deposits = rasterio.open(deposits)
        else:
            deposits = gpd.read_file(deposits)

    if arrays_to_generate == []:
        arrays_to_generate = None

    with ProgressLog.running_algorithm():
        df, arrays, raster_meta, nr_of_deposits, nr_of_pixels = weights_of_evidence_calculate_weights(
            evidential_raster=evidential_raster,
            deposits=deposits,
            raster_nodata=raster_nodata,
            weights_type=weights_type,
            studentized_contrast_threshold=studentized_contrast_threshold,
            arrays_to_generate=arrays_to_generate,
        )

    with ProgressLog.saving_output_files([output_raster_dir, output_results_table]):
        df.to_csv(output_results_table)

        out_rasters_dict = {}
        file_name = evidential_raster.name.split("/")[-1].split(".")[0]
        raster_meta.pop("dtype")  # Remove dtype from metadata to set it individually

        for key, array in arrays.items():
            # Set correct dtype for the array
            if key in ["Class", "Pixel count", "Deposit count"]:
                dtype = np.uint8
            else:
                dtype = np.float32

            array = nan_to_nodata(array, raster_meta["nodata"])
            output_raster_name = file_name + "_weights_" + weights_type + "_" + key
            output_raster_path = output_raster_dir.joinpath(output_raster_name + ".tif")
            with rasterio.open(output_raster_path, "w", dtype=dtype, **raster_meta) as dst:
                dst.write(array, 1)
            out_rasters_dict[output_raster_name] = str(output_raster_path)

    ResultSender.send_dict_as_json(out_rasters_dict)
    ProgressLog.finish()


@app.command()
def weights_of_evidence_calculate_responses_cli(
    input_rasters_weights: INPUT_FILES_OPTION,
    input_rasters_standard_deviations: INPUT_FILES_OPTION,
    input_weights_table: INPUT_FILE_OPTION,
    output_probabilities: OUTPUT_FILE_OPTION,
    output_probabilities_std: OUTPUT_FILE_OPTION,
    output_confidence_array: OUTPUT_FILE_OPTION,
):
    """
    Calculate the posterior probabilities for the given generalized weight arrays.

    Parameter --input-rasters are the output arrays (rasters) of weights-of-evidence-calculate-weights-cli.
    For each set of rasters, generalized weight and generalized standard deviation arrays are used and summed
    together pixel-wise to calculate the posterior probabilities. If generalized arrays are not found,
    the W+ and S_W+ arrays are used (so if outputs from unique weight calculations are used for this function).
    """
    from eis_toolkit.prediction.weights_of_evidence import weights_of_evidence_calculate_responses

    with ProgressLog.reading_input_files():
        typer.echo(input_rasters_weights)

        dict_array = []
        raster_profile = None

        for raster_weights, raster_std in zip_longest(
            input_rasters_weights, input_rasters_standard_deviations, fillvalue=None
        ):

            if raster_weights is not None:
                with rasterio.open(raster_weights) as src:
                    array_W = src.read(1)
                    array_W = nodata_to_nan(array_W, src.nodata)

                    if raster_profile is None:
                        raster_profile = src.profile

            if raster_std is not None:
                with rasterio.open(raster_std) as src:
                    array_S_W = src.read(1)
                    array_S_W = nodata_to_nan(array_S_W, src.nodata)

            dict_array.append({"W+": array_W, "S_W+": array_S_W})

        weights_df = pd.read_csv(input_weights_table)

    with ProgressLog.running_algorithm():
        posterior_probabilities, posterior_probabilies_std, confidence_array = weights_of_evidence_calculate_responses(
            output_arrays=dict_array, weights_df=weights_df
        )

    with ProgressLog.saving_output_files([output_probabilities, output_probabilities_std, output_confidence_array]):
        posterior_probabilities = nan_to_nodata(posterior_probabilities, raster_profile["nodata"])
        with rasterio.open(output_probabilities, "w", **raster_profile) as dst:
            dst.write(posterior_probabilities, 1)

        posterior_probabilies_std = nan_to_nodata(posterior_probabilies_std, raster_profile["nodata"])
        with rasterio.open(output_probabilities_std, "w", **raster_profile) as dst:
            dst.write(posterior_probabilies_std, 1)

        confidence_array = nan_to_nodata(confidence_array, raster_profile["nodata"])
        with rasterio.open(output_confidence_array, "w", **raster_profile) as dst:
            dst.write(confidence_array, 1)

    ProgressLog.finish()


@app.command()
def agterberg_cheng_CI_test_cli(
    input_posterior_probabilities: INPUT_FILE_OPTION,
    input_posterior_probabilities_std: INPUT_FILE_OPTION,
    input_weights_table: INPUT_FILE_OPTION,
):
    """Perform the conditional independence test presented by Agterberg-Cheng (2002)."""
    from eis_toolkit.prediction.weights_of_evidence import agterberg_cheng_CI_test

    with ProgressLog.reading_input_files():
        with rasterio.open(input_posterior_probabilities) as src:
            posterior_probabilities = src.read(1)
            posterior_probabilities = nodata_to_nan(posterior_probabilities, src.nodata)

        with rasterio.open(input_posterior_probabilities_std) as src:
            posterior_probabilities_std = src.read(1)
            posterior_probabilities_std = nodata_to_nan(posterior_probabilities_std, src.nodata)

        weights_df = pd.read_csv(input_weights_table)

    with ProgressLog.running_algorithm():
        _, _, _, _, summary = agterberg_cheng_CI_test(
            posterior_probabilities=posterior_probabilities,
            posterior_probabilities_std=posterior_probabilities_std,
            weights_df=weights_df,
        )

    typer.echo(summary)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_df = alr_transform(df=df, column=column, keep_denominator_column=keep_denominator_column)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_df = inverse_alr(df=df, denominator_column=denominator_column, scale=scale)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


# CODA - CLR TRANSFORM
@app.command()
def clr_transform_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Perform a centered logratio transformation on the data."""
    from eis_toolkit.transformations.coda.clr import clr_transform

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_df = clr_transform(df=df)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_df = inverse_clr(df=df, colnames=colnames, scale=scale)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_series = single_ilr_transform(df=df, subcomposition_1=subcomposition_1, subcomposition_2=subcomposition_2)

    with ProgressLog.saving_output_files(output_vector):
        # NOTE: Output of pairwise_logratio might be changed to DF in the future, to automatically do the following
        df["single_ilr"] = out_series
        out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_series = pairwise_logratio(df=df, numerator_column=numerator_column, denominator_column=denominator_column)

    with ProgressLog.saving_output_files(output_vector):
        # NOTE: Output of pairwise_logratio might be changed to DF in the future, to automatically do the following
        df["pairwise_logratio"] = out_series
        out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


# CODA - SINGLE PLR TRANSFORM
@app.command()
def single_plr_transform_cli(
    input_vector: INPUT_FILE_OPTION,
    output_vector: OUTPUT_FILE_OPTION,
    column: str = typer.Option(),
):
    """Perform a pivot logratio transformation on the selected column."""
    from eis_toolkit.transformations.coda.plr import single_plr_transform

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_series = single_plr_transform(df=df, column=column)

    with ProgressLog.saving_output_files(output_vector):
        # NOTE: Output of single_plr_transform might be changed to DF in the future, to automatically do the following
        df["single_plr"] = out_series
        out_gdf = gpd.GeoDataFrame(df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


# CODA - PLR TRANSFORM
@app.command()
def plr_transform_cli(input_vector: INPUT_FILE_OPTION, output_vector: OUTPUT_FILE_OPTION):
    """Perform a pivot logratio transformation on the dataframe, returning the full set of transforms."""
    from eis_toolkit.transformations.coda.plr import plr_transform

    with ProgressLog.reading_input_files():
        gdf = gpd.read_file(input_vector)
        geometries = gdf["geometry"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    with ProgressLog.running_algorithm():
        out_df = plr_transform(df=df)

    with ProgressLog.saving_output_files(output_vector):
        out_gdf = gpd.GeoDataFrame(out_df, geometry=geometries)
        out_gdf.to_file(output_vector)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = binarize(raster=raster, thresholds=[threshold])
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = clip_transform(raster=raster, limits=[(limit_lower, limit_higher)])
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = z_score_normalization(raster=raster)
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = min_max_scaling(raster=raster, new_range=[(min, max)])
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = log_transform(raster=raster, log_transform=[get_enum_values(log_type)])
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = sigmoid_transform(
            raster=raster, bounds=[(limit_lower, limit_upper)], slope=[slope], center=center
        )
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta, _ = winsorize(
            raster=raster, percentiles=[(percentile_lower, percentile_higher)], inside=inside
        )
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


# ---EVALUATION ---


@app.command()
def summarize_probability_metrics_cli(true_labels: INPUT_FILE_OPTION, probabilities: INPUT_FILE_OPTION):
    """
    Generate a comprehensive report of various evaluation metrics for classification probabilities.

    The output includes ROC AUC, log loss, average precision and Brier score loss.
    """
    from eis_toolkit.evaluation.classification_probability_evaluation import summarize_probability_metrics
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    with ProgressLog.reading_input_files():
        (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])

    with ProgressLog.running_algorithm():
        results_dict = summarize_probability_metrics(y_true=y_true, y_prob=y_prob, decimals=3)

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


@app.command()
def summarize_label_metrics_binary_cli(true_labels: INPUT_FILE_OPTION, predictions: INPUT_FILE_OPTION):
    """
    Generate a comprehensive report of various evaluation metrics for binary classification results.

    The output includes accuracy, precision, recall, F1 scores and confusion matrix elements
    (true negatives, false positives, false negatives, true positives).
    """
    from eis_toolkit.evaluation.classification_label_evaluation import summarize_label_metrics_binary
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    with ProgressLog.reading_input_files():
        (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])

    with ProgressLog.running_algorithm():
        results_dict = summarize_label_metrics_binary(y_true=y_true, y_pred=y_pred, decimals=3)

    ResultSender.send_dict_as_json(results_dict)
    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])

    with ProgressLog.running_algorithm():
        _ = plot_roc_curve(y_true=y_true, y_prob=y_prob)

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])

    with ProgressLog.running_algorithm():
        _ = plot_det_curve(y_true=y_true, y_prob=y_prob)

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])

    with ProgressLog.running_algorithm():
        _ = plot_precision_recall_curve(y_true=y_true, y_prob=y_prob)

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        (y_prob, y_true), _, _ = read_data_for_evaluation([probabilities, true_labels])

    with ProgressLog.running_algorithm():
        _ = plot_calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=n_bins)

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


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

    with ProgressLog.reading_input_files():
        (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])

    with ProgressLog.running_algorithm():
        matrix = confusion_matrix(y_true, y_pred)
        _ = plot_confusion_matrix(confusion_matrix=matrix)

    if output_file is not None:
        with ProgressLog.saving_output_files(output_file):
            dpi = "figure" if save_dpi is None else save_dpi
            plt.savefig(output_file, dpi=dpi)

    if show_plot:
        plt.show()

    ProgressLog.finish()


@app.command()
def score_predictions_cli(
    true_labels: INPUT_FILE_OPTION,
    predictions: INPUT_FILE_OPTION,
    metrics: Annotated[List[str], typer.Option()],
    decimals: Optional[int] = None,
):
    """Score predictions."""
    from eis_toolkit.evaluation.scoring import score_predictions
    from eis_toolkit.prediction.machine_learning_general import read_data_for_evaluation

    with ProgressLog.reading_input_files():
        (y_pred, y_true), _, _ = read_data_for_evaluation([predictions, true_labels])

    with ProgressLog.running_algorithm():
        outputs = score_predictions(y_true, y_pred, metrics, decimals)

    ResultSender.send_dict_as_json(outputs)
    ProgressLog.finish()


# --- UTILITIES ---
@app.command()
def split_raster_bands_cli(input_raster: INPUT_FILE_OPTION, output_dir: OUTPUT_DIR_OPTION):  # CHECK
    """Split multiband raster into singleband rasters."""
    from eis_toolkit.utilities.file_io import get_output_paths_from_common_name
    from eis_toolkit.utilities.raster import split_raster_bands

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        output_singleband_rasters = split_raster_bands(raster)
    raster.close()

    with ProgressLog.saving_output_files(output_dir):
        name = os.path.splitext(os.path.basename(input_raster))[0]
        output_paths = get_output_paths_from_common_name(output_singleband_rasters, output_dir, f"{name}_split", ".tif")
        for output_path, (out_image, out_profile) in zip(output_paths, output_singleband_rasters):
            with rasterio.open(output_path, "w", **out_profile) as dst:
                dst.write(out_image, 1)

    ProgressLog.finish()


@app.command()
def combine_raster_bands_cli(input_rasters: INPUT_FILES_ARGUMENT, output_raster: OUTPUT_FILE_OPTION):
    """Combine multiple rasters into one multiband raster."""
    from eis_toolkit.utilities.raster import combine_raster_bands

    with ProgressLog.reading_input_files():
        rasters = [rasterio.open(raster) for raster in input_rasters]  # Open all rasters to be combined

    with ProgressLog.running_algorithm():
        out_image, out_meta = combine_raster_bands(rasters)
    [raster.close() for raster in rasters]

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


@app.command()
def unify_raster_nodata_cli(
    input_rasters: INPUT_FILES_ARGUMENT, output_dir: OUTPUT_DIR_OPTION, new_nodata: float = -9999  # CHECK
):
    """Unifies nodata for the input rasters."""
    from eis_toolkit.utilities.file_io import get_output_paths_from_inputs
    from eis_toolkit.utilities.nodata import unify_raster_nodata

    with ProgressLog.reading_input_files():
        rasters = [rasterio.open(raster) for raster in input_rasters]  # Open all rasters to be unified

    with ProgressLog.running_algorithm():
        unified = unify_raster_nodata(rasters, new_nodata)
    [raster.close() for raster in rasters]

    with ProgressLog.saving_output_files(output_dir):
        output_paths = get_output_paths_from_inputs(input_rasters, output_dir, "nodata_unified", ".tif")
        for output_path, (out_image, out_profile) in zip(output_paths, unified):
            with rasterio.open(output_path, "w", **out_profile) as dst:
                dst.write(out_image)

    ProgressLog.finish()


@app.command()
def convert_raster_nodata_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    old_nodata: float = None,
    new_nodata: float = -9999,
):
    """Convert existing nodata values with a new nodata value for a raster."""
    from eis_toolkit.utilities.nodata import convert_raster_nodata

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = convert_raster_nodata(raster, old_nodata, new_nodata)
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


@app.command()
def replace_with_nodata_cli(
    input_raster: INPUT_FILE_OPTION,
    output_raster: OUTPUT_FILE_OPTION,
    target_value: Annotated[float, typer.Option()],
    nodata_value: float = None,
    replace_condition: Annotated[ReplaceCondition, typer.Option(case_sensitive=False)] = ReplaceCondition.equal,
):
    """Replace raster pixel values with nodata."""
    from eis_toolkit.utilities.nodata import replace_with_nodata

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)

    with ProgressLog.running_algorithm():
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


@app.command()
def set_raster_nodata_cli(
    input_raster: INPUT_FILE_OPTION, output_raster: OUTPUT_FILE_OPTION, new_nodata: float = typer.Option()
):
    """Set new nodata value for raster profile."""
    from eis_toolkit.utilities.nodata import set_raster_nodata

    with ProgressLog.reading_input_files():
        raster = rasterio.open(input_raster)
        out_image = raster.read()

    with ProgressLog.running_algorithm():
        out_meta = set_raster_nodata(raster.meta, new_nodata)
    raster.close()

    with ProgressLog.saving_output_files(output_raster):
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    ProgressLog.finish()


# if __name__ == "__main__":
def cli():
    """CLI app."""
    app()


if __name__ == "__main__":
    app()
