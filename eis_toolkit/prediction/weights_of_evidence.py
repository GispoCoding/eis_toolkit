from numbers import Number

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import List, Literal, Optional, Sequence, Tuple

from eis_toolkit import exceptions
from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

CLASS_COLUMN = "Class"
PIXEL_COUNT_COLUMN = "Pixel count"
DEPOSIT_COUNT_COLUMN = "Deposit count"
WEIGHT_PLUS_COLUMN = "W+"
WEIGHT_S_PLUS_COLUMN = "S_W+"
WEIGHT_MINUS_COLUMN = "W-"
WEIGHT_S_MINUS_COLUMN = "S_W-"
CONTRAST_COLUMN = "Contrast"
S_CONTRAST_COLUMN = "S_Contrast"
STUDENTIZED_CONTRAST_COLUMN = "Studentized contrast"
GENERALIZED_CLASS_COLUMN = "Generalized class"
GENERALIZED_WEIGHT_PLUS_COLUMN = "Generalized W+"
GENERALIZED_S_WEIGHT_PLUS_COLUMN = "Generalized S_W+"

VALID_DF_COLUMNS = [
    CLASS_COLUMN,
    PIXEL_COUNT_COLUMN,
    DEPOSIT_COUNT_COLUMN,
    WEIGHT_PLUS_COLUMN,
    WEIGHT_S_PLUS_COLUMN,
    WEIGHT_MINUS_COLUMN,
    WEIGHT_S_MINUS_COLUMN,
    CONTRAST_COLUMN,
    S_CONTRAST_COLUMN,
    STUDENTIZED_CONTRAST_COLUMN,
    GENERALIZED_CLASS_COLUMN,
    GENERALIZED_WEIGHT_PLUS_COLUMN,
    GENERALIZED_S_WEIGHT_PLUS_COLUMN,
]

DEFAULT_METRICS_UNIQUE = [CLASS_COLUMN, WEIGHT_PLUS_COLUMN, WEIGHT_S_PLUS_COLUMN]
DEFAULT_METRICS_CUMULATIVE = [
    CLASS_COLUMN,
    WEIGHT_PLUS_COLUMN,
    WEIGHT_S_PLUS_COLUMN,
    GENERALIZED_WEIGHT_PLUS_COLUMN,
    GENERALIZED_S_WEIGHT_PLUS_COLUMN,
]


def _read_and_preprocess_evidence(
    raster: rasterio.io.DatasetReader, nodata: Optional[Number] = None, band: int = 1
) -> np.ndarray:
    """Read raster data and handle NoData values."""

    array = np.array(raster.read(band), dtype=np.float32)

    if nodata is not None:
        array[array == nodata] = np.nan
    elif raster.meta["nodata"] is not None:
        array[array == raster.meta["nodata"]] = np.nan

    return array


def _calculate_metrics_for_class(
    deposits: np.ndarray, evidence: np.ndarray
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Calculate weights/metrics for given data."""
    A = np.sum(np.logical_and(deposits == 1, evidence == 1))
    B = np.sum(np.logical_and(deposits == 1, evidence == 0))
    C = np.sum(np.logical_and(deposits == 0, evidence == 1))
    D = np.sum(np.logical_and(deposits == 0, evidence == 0))

    # If data has no deposits or every evidence pixel has a deposit
    if A == 0 or C + D == 0:
        return A, B, C, D, 0, 0, 0, 0, 0, 0, 0

    p_A_nominator = A
    p_C_nominator = C
    B_adjusted = B
    D_adjusted = D

    if B == 0:
        p_A_nominator -= 0.99
        B_adjusted = 0.99

    if D == 0:
        p_C_nominator -= 0.99
        D_adjusted = 0.99

    p_A = p_A_nominator / (A + B)  # probability of presence of evidence given the presence of mineral deposit
    p_C = p_C_nominator / (C + D)  # probability of presence of evidence given the absence of mineral deposit

    # Calculate metrics
    w_plus = np.log(p_A / p_C) if p_C != 0 else 0  # Check
    w_minus = np.log((1 - p_A) / (1 - p_C))
    contrast = w_plus - w_minus

    # Calculate signifigance metrics
    s_w_plus = np.sqrt((1 / p_A_nominator) + (1 / p_C_nominator))
    s_w_minus = np.sqrt((1 / B_adjusted) + (1 / D_adjusted))

    s_contrast = np.sqrt(s_w_plus**2 + s_w_minus**2)
    studentized_contrast = contrast / s_contrast

    return A, B, C, D, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast, studentized_contrast


def _unique_weights(deposits: np.ndarray, evidence: np.ndarray) -> dict:
    """Calculate unique weights for each class."""
    classes = np.unique(evidence)
    return {cls: _calculate_metrics_for_class(deposits, evidence == cls) for cls in classes}


def _cumulative_weights(deposits: np.ndarray, evidence: np.ndarray, ascending: bool = True) -> dict:
    """Calculate cumulative weights (ascending or descending) for each class."""
    classes = sorted(np.unique(evidence), reverse=not ascending)
    cumulative_classes = [classes[: i + 1] for i in range(len(classes))]
    return {
        cls[i]: _calculate_metrics_for_class(deposits, np.isin(evidence, cls))
        for i, cls in enumerate(cumulative_classes)
    }


def _generalized_classes_categorical(df: pd.DataFrame, studentized_contrast_threshold: Number) -> pd.DataFrame:
    gen_df = df.copy()
    gen_df[GENERALIZED_CLASS_COLUMN] = gen_df[CLASS_COLUMN]

    reclassified = False
    for i in range(0, len(gen_df.index)):
        if abs(gen_df.loc[i, STUDENTIZED_CONTRAST_COLUMN]) < studentized_contrast_threshold:
            gen_df.loc[i, GENERALIZED_CLASS_COLUMN] = 99
            reclassified = True

    if not reclassified:
        raise exceptions.ClassificationFailedException(
            "Failed to create generalized classes with given studentized contrast treshold ({})".format(
                studentized_contrast_threshold
            )
        )

    gen_df = gen_df.sort_values(by=GENERALIZED_CLASS_COLUMN, ascending=True)

    return gen_df


def _calculate_generalized_weights_categorical(df: pd.DataFrame, deposits) -> pd.DataFrame:
    gen_df = df.copy()
    total_deposits = np.sum(deposits == 1)
    total_no_deposits = deposits.size - total_deposits

    # Class 99 (gen class)
    class_99_count = 0
    class_99_point_count = 0

    for i in range(0, len(gen_df.index)):
        if gen_df.loc[i, GENERALIZED_CLASS_COLUMN] == 99:
            # class_99_count = max(gen_df.loc[i, PIXEL_COUNT_COLUMN], class_99_count)
            # class_99_point_count = max(gen_df.loc[i, DEPOSIT_COUNT_COLUMN], class_99_point_count)
            class_99_count += gen_df.loc[i, PIXEL_COUNT_COLUMN]
            class_99_point_count += gen_df.loc[i, DEPOSIT_COUNT_COLUMN]

    class_99_w_gen = np.log(class_99_point_count / total_deposits) - np.log(
        (class_99_count - class_99_point_count) / total_no_deposits
    )
    clas_99_s_wpls_gen = np.sqrt((1 / class_99_point_count) + (1 / (class_99_count - class_99_point_count)))

    gen_df[GENERALIZED_WEIGHT_PLUS_COLUMN] = gen_df[WEIGHT_PLUS_COLUMN]
    gen_df[GENERALIZED_S_WEIGHT_PLUS_COLUMN] = gen_df[WEIGHT_S_PLUS_COLUMN]

    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 99, GENERALIZED_WEIGHT_PLUS_COLUMN] = round(class_99_w_gen, 4)
    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 99, GENERALIZED_S_WEIGHT_PLUS_COLUMN] = round(clas_99_s_wpls_gen, 4)

    return gen_df


def _generalized_classes_cumulative(df: pd.DataFrame, studentized_contrast_threshold: Number) -> pd.DataFrame:
    """Create generalized classes based on contrast and studentized contrast threhsold value."""
    gen_df = df.copy()
    index = gen_df.idxmax()[CONTRAST_COLUMN]

    if (
        gen_df.loc[index, STUDENTIZED_CONTRAST_COLUMN] < studentized_contrast_threshold
        or index == len(gen_df.index) - 1
    ):
        raise exceptions.ClassificationFailedException(
            "Failed to create generalized classes with given studentized contrast treshold ({} < {})".format(
                gen_df.loc[index, STUDENTIZED_CONTRAST_COLUMN], studentized_contrast_threshold
            )
        )

    gen_df[GENERALIZED_CLASS_COLUMN] = 1
    for i in range(0, index + 1):
        gen_df.loc[i, GENERALIZED_CLASS_COLUMN] = 2

    return gen_df


def _calculate_generalized_weights_cumulative(df: pd.DataFrame, deposits) -> pd.DataFrame:
    """Calculate generalized weights.for cumulative methods."""
    gen_df = df.copy()
    total_deposits = np.sum(deposits == 1)
    total_no_deposits = deposits.size - total_deposits

    # Class 2
    class_2_max_index = gen_df.idxmax()[CONTRAST_COLUMN]
    class_2_count = gen_df.loc[class_2_max_index, PIXEL_COUNT_COLUMN]
    class_2_point_count = gen_df.loc[class_2_max_index, DEPOSIT_COUNT_COLUMN]

    class_2_w_gen = np.log(class_2_point_count / total_deposits) - np.log(
        (class_2_count - class_2_point_count) / total_no_deposits
    )
    clas_2_s_wpls_gen = np.sqrt((1 / class_2_point_count) + (1 / (class_2_count - class_2_point_count)))

    gen_df[GENERALIZED_WEIGHT_PLUS_COLUMN] = round(class_2_w_gen, 4)
    gen_df[GENERALIZED_S_WEIGHT_PLUS_COLUMN] = round(clas_2_s_wpls_gen, 4)

    # Class 1
    class_1_count = gen_df.loc[len(gen_df.index) - 1, PIXEL_COUNT_COLUMN] - class_2_count
    class_1_point_count = gen_df.loc[len(gen_df.index) - 1, DEPOSIT_COUNT_COLUMN] - class_2_point_count

    class_1_w_gen = np.log(class_1_point_count / total_deposits) - np.log(
        (class_1_count - class_1_point_count) / total_no_deposits
    )
    clas_1_s_wpls_gen = np.sqrt((1 / class_1_point_count) + (1 / (class_1_count - class_1_point_count)))
    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 1, GENERALIZED_WEIGHT_PLUS_COLUMN] = round(class_1_w_gen, 4)
    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 1, GENERALIZED_S_WEIGHT_PLUS_COLUMN] = round(clas_1_s_wpls_gen, 4)

    return gen_df


def _generate_rasters_from_metrics(evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str]) -> dict:
    """Generate rasters for defined metrics based."""
    raster_dict = {}
    for metric in metrics_to_include:
        raster = np.full(evidence.shape, np.nan)
        for _, row in df.iterrows():
            mask = np.isin(evidence, row[CLASS_COLUMN])
            raster[mask] = row[metric]
        raster_dict[metric] = raster
    return raster_dict


@beartype
def weights_of_evidence(
    evidential_raster: rasterio.io.DatasetReader,
    deposits: gpd.GeoDataFrame,
    raster_nodata: Optional[Number] = None,
    weights_type: Literal["unique", "categorical", "ascending", "descending"] = "unique",
    studentized_contrast_threshold: Number = 1,
    rasters_to_generate: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Calculate weights of spatial associations.

    Args:
        evidential_raster: The evidential raster.
        deposits: Vector data representing the mineral deposits or occurences point data.
        raster_nodata: If nodata value of raster is wanted to specify manually. Optional parameter, defaults to None
            (nodata from raster metadata is used).
        weights_type: Accepted values are 'unique', 'categorical', 'ascending' and 'descending'.
            Unique weights does not create generalized classes and does not use a studentized contrast threshold value
            while categorical, cumulative ascending and cumulative descending do. Categorical weights are calculated so
            that all classes with studentized contrast below the defined threshold are grouped into one generalized
            class. Cumulative ascending and descending weights find the class with max contrast and group classes
            above/below into generalized classes. Generalized weights are also calculated for generalized classes.
        studentized_contrast_threshold: Studentized contrast threshold value used with 'categorical', 'ascending' and
            'descending' weight types. Used either as reclassification threshold directly (categorical) or to check
            that class with max contrast has studentized contrast value at least the defined value (cumulative).
            Defaults to 1.
        rasters_to_generate: Rasters to generate from the computed weight metrics. All column names
            in the produced weights_df are valid choices. Defaults to ["Class", "W+", "S_W+]
            for "unique" weights_type and ["Class", "W+", "S_W+", "Generalized W+", "Generalized S_W+"]
            for the cumulative weight types.

    Returns:
        Dataframe with weights of spatial association between the input data.
        Dictionary of rasters for specified metrics.
        Raster metadata.
    """

    if rasters_to_generate is None:
        if weights_type == "unique":
            metrics_to_rasters = DEFAULT_METRICS_UNIQUE
        else:
            metrics_to_rasters = DEFAULT_METRICS_CUMULATIVE
    else:
        for col_name in rasters_to_generate:
            if col_name not in VALID_DF_COLUMNS:
                raise exceptions.InvalidColumnException(
                    f"Rasters to generate contains invalid metric / column name: {col_name}."
                )
        metrics_to_rasters = rasters_to_generate.copy()

    # 1. Preprocess data
    evidence_array = _read_and_preprocess_evidence(evidential_raster, raster_nodata)
    raster_meta = evidential_raster.meta

    # Rasterize deposits
    deposit_array, _ = rasterize_vector(
        geodataframe=deposits, default_value=1.0, base_raster_profile=raster_meta, fill_value=0.0
    )

    # Mask NaN out of the array
    nodata_mask = np.isnan(evidence_array)
    masked_evidence_array = evidence_array[~nodata_mask]
    masked_deposit_array = deposit_array[~nodata_mask]

    # 2. WofE calculations
    if weights_type == "unique" or weights_type == "categorical":
        wofe_weights = _unique_weights(masked_deposit_array, masked_evidence_array)
    elif weights_type == "ascending":
        wofe_weights = _cumulative_weights(masked_deposit_array, masked_evidence_array, ascending=True)
    elif weights_type == "descending":
        wofe_weights = _cumulative_weights(masked_deposit_array, masked_evidence_array, ascending=False)
    else:
        raise exceptions.InvalidParameterValueException(
            "Expected weights_type to be one of unique, categorical, ascending or descending."
        )

    # 3. Create DataFrame based on calculated metrics
    df_entries = []
    for cls, metrics in wofe_weights.items():
        metrics = [round(metric, 4) if isinstance(metric, np.floating) else metric for metric in metrics]
        A, _, C, _, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast, studentized_contrast = metrics
        df_entries.append(
            {
                CLASS_COLUMN: cls,
                PIXEL_COUNT_COLUMN: A + C,
                DEPOSIT_COUNT_COLUMN: A,
                WEIGHT_PLUS_COLUMN: w_plus,
                WEIGHT_S_PLUS_COLUMN: s_w_plus,
                WEIGHT_MINUS_COLUMN: w_minus,
                WEIGHT_S_MINUS_COLUMN: s_w_minus,
                CONTRAST_COLUMN: contrast,
                S_CONTRAST_COLUMN: s_contrast,
                STUDENTIZED_CONTRAST_COLUMN: studentized_contrast,
            }
        )
    weights_df = pd.DataFrame(df_entries)

    # 4. If we use cumulative weights type, calculate generalized classes and weights
    if weights_type == "categorical":
        weights_df = _generalized_classes_categorical(weights_df, studentized_contrast_threshold)
        weights_df = _calculate_generalized_weights_categorical(weights_df, masked_deposit_array)
    elif weights_type == "ascending" or weights_type == "descending":
        weights_df = _generalized_classes_cumulative(weights_df, studentized_contrast_threshold)
        weights_df = _calculate_generalized_weights_cumulative(weights_df, masked_deposit_array)

    # 5. Generate rasters for desired metrics
    raster_dict = _generate_rasters_from_metrics(evidence_array, weights_df, metrics_to_rasters)

    return weights_df, raster_dict, raster_meta
