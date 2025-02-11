import warnings
from numbers import Number

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from eis_toolkit.exceptions import (
    ClassificationFailedException,
    InvalidColumnException,
    InvalidParameterValueException,
    NonMatchingRasterMetadataException,
)
from eis_toolkit.utilities.checks.raster import check_raster_grids
from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector
from eis_toolkit.warnings import ClassificationFailedWarning, InvalidColumnWarning

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

GENERALIZED_COLUMNS = [GENERALIZED_CLASS_COLUMN, GENERALIZED_WEIGHT_PLUS_COLUMN, GENERALIZED_S_WEIGHT_PLUS_COLUMN]
WEIGHTS_COLUMNS = [
    WEIGHT_PLUS_COLUMN,
    WEIGHT_S_PLUS_COLUMN,
    WEIGHT_MINUS_COLUMN,
    WEIGHT_S_MINUS_COLUMN,
]

REQUIRED_FOR_GENERALIZATION = {
    "manual": [],
    "max_contrast": [CONTRAST_COLUMN],
    "max_contrast_if_feasible": [CONTRAST_COLUMN, STUDENTIZED_CONTRAST_COLUMN],
    "max_feasible_contrast": [CONTRAST_COLUMN, STUDENTIZED_CONTRAST_COLUMN],
    "max_studentized_contrast": [STUDENTIZED_CONTRAST_COLUMN],
}


def _read_and_preprocess_raster_data(
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
        raise ClassificationFailedException(
            "Failed to create generalized classes with given studentized contrast treshold ({})".format(
                studentized_contrast_threshold
            )
        )

    gen_df = gen_df.sort_values(by=GENERALIZED_CLASS_COLUMN, ascending=True)

    return gen_df


def _generalized_weights_categorical(df: pd.DataFrame, deposits) -> pd.DataFrame:
    """Calculate generalized weights for categorical weights type. Assumes class 99 exists as the general class."""
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


def _generalized_classes_cumulative(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """Create generalized classes based on given index for cutoff row."""
    gen_df = df.copy()

    gen_df[GENERALIZED_CLASS_COLUMN] = 1
    for i in range(0, index + 1):
        gen_df.loc[i, GENERALIZED_CLASS_COLUMN] = 2

    return gen_df


def _generalized_weights_cumulative(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """
    Calculate generalized weights for cumulative methods.

    Assumes there are classes 1 and 2 as the general classes.
    """
    gen_df = df.copy()

    # Class 2
    gen_df[GENERALIZED_WEIGHT_PLUS_COLUMN] = gen_df.loc[index, WEIGHT_PLUS_COLUMN]
    gen_df[GENERALIZED_S_WEIGHT_PLUS_COLUMN] = gen_df.loc[index, WEIGHT_S_PLUS_COLUMN]

    # Class 1
    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 1, GENERALIZED_WEIGHT_PLUS_COLUMN] = gen_df.loc[
        index, WEIGHT_MINUS_COLUMN
    ]
    gen_df.loc[gen_df[GENERALIZED_CLASS_COLUMN] == 1, GENERALIZED_S_WEIGHT_PLUS_COLUMN] = gen_df.loc[
        index, WEIGHT_S_MINUS_COLUMN
    ]

    return gen_df


def _generate_arrays_from_metrics(
    evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str]
) -> Dict[str, np.ndarray]:
    """Generate arrays for defined metrics."""
    array_dict = {}
    for metric in metrics_to_include:
        metric_array = np.full(evidence.shape, np.nan)
        for _, row in df.iterrows():
            mask = np.isin(evidence, row[CLASS_COLUMN])
            metric_array[mask] = row[metric]
        array_dict[metric] = metric_array
    return array_dict


def _calculate_nr_of_deposit_pixels(array: np.ndarray, df: pd.DataFrame) -> Tuple[int, int]:
    masked_array = array[~np.isnan(array)]
    nr_of_pixels = int(np.size(masked_array))

    pixels_column = df["Pixel count"]

    match = pixels_column == nr_of_pixels
    if match.any():
        nr_of_deposits = df.loc[match, "Deposit count"].iloc[0]
    else:
        nr_of_pixels = df["Pixel count"].sum()
        nr_of_deposits = df["Deposit count"].sum()

    return nr_of_deposits, nr_of_pixels


@beartype
def generalize_weights_cumulative(
    df: pd.DataFrame,
    classification_method: Literal[
        "manual", "max_contrast", "max_contrast_if_feasible", "max_feasible_contrast", "max_studentized_contrast"
    ] = "max_contrast_if_feasible",
    manual_cutoff_index: Optional[Number] = None,
    studentized_contrast_threshold: Optional[Number] = 1,
) -> pd.DataFrame:
    """
    Calculate generalized weights for cumulative methods.

    Perform binary reclassification into the generalized classes 1 and 2 according to the selected
    classification method. Calculate generalized weights for the two classes.

    Args:
        df: A weights table returned by weights_of_evidence_calculate_weights.
        classification_method: Accepted values are 'manual', 'max_contrast',
            max_contrast_if_feasible, 'max_feasible_contrast' and 'max_studentized_contrast', detailed below:
            'manual': Requires a valid row index to use as cutoff value.
            'max_contrast': Uses the maximum contrast value regardless of studentized contrast.
            'max_contrast_if_feasible': Uses the maximum contrast value if the corresponding studentized
                contrast is greater than the provided threshold value.
            'max_feasible_contrast': Uses the highest contrast value for which the studentized contrast
                is greater than the provided threshold value.
            'max_studentized_contrast': Uses the highest studentized contrast value.
        manual_cutoff_index: Index of the last row to be included in class 2.
        studentized_contrast_threshold: Studentized contrast threshold value used to check that class with
            max contrast has studentized contrast value at least the defined value. Defaults to 1.
    Returns:
        The weights table with the addition of a generalized class column. If generalization failed, returns
            the original table.
    Warns:
        ClassificationFailedWarning
        InvalidColumnWarning
    """
    df = df.copy()

    required_columns = WEIGHTS_COLUMNS + REQUIRED_FOR_GENERALIZATION[classification_method]
    missing_columns = [col for col in required_columns if col not in df.columns.values]

    if len(missing_columns) != 0:
        warnings.warn(
            f"Failed to create generalized classes. The following columns are required: {missing_columns}",
            InvalidColumnWarning,
        )
        return df

    columns_to_drop = [col for col in df.columns.values if col in GENERALIZED_COLUMNS]
    df = df.drop(columns_to_drop, axis=1)

    index = len(df.index) - 1
    classification_failed_warning = ""

    if classification_method == "manual":
        if manual_cutoff_index is None:
            classification_failed_warning = f"Failed to create generalized classes with the row index \
                {manual_cutoff_index}"
        else:
            index = manual_cutoff_index

    elif classification_method == "max_contrast":
        index = df.idxmax()[CONTRAST_COLUMN]

    elif classification_method == "max_contrast_if_feasible":

        index = df.idxmax()[CONTRAST_COLUMN]

        if df.loc[index, STUDENTIZED_CONTRAST_COLUMN] < studentized_contrast_threshold:
            classification_failed_warning = f"Failed to create generalized classes with given studentized \
            contrast threshold {studentized_contrast_threshold}"

    elif classification_method == "max_feasible_contrast":
        df_studentized_contrast = df[df[STUDENTIZED_CONTRAST_COLUMN] > studentized_contrast_threshold]

        if len(df_studentized_contrast) == 0:
            classification_failed_warning = f"Failed to create generalized classes with given studentized \
                contrast threshold {studentized_contrast_threshold}"
        else:
            index = df_studentized_contrast.idxmax()[CONTRAST_COLUMN]

    else:
        # max_studentized_contrast
        index = df.idxmax()[STUDENTIZED_CONTRAST_COLUMN]

    if classification_failed_warning != "":
        warnings.warn(
            classification_failed_warning,
            ClassificationFailedWarning,
        )
        return df

    if index >= len(df.index) - 1:
        warnings.warn("Failed to create generalized classes.", ClassificationFailedWarning)

        return df
    else:
        df = _generalized_classes_cumulative(df, index)

        return _generalized_weights_cumulative(df, index)


@beartype
def weights_of_evidence_calculate_weights(
    evidential_raster: rasterio.io.DatasetReader,
    deposits: Union[gpd.GeoDataFrame, rasterio.io.DatasetReader],
    raster_nodata: Optional[Number] = None,
    weights_type: Literal["unique", "categorical", "ascending", "descending"] = "unique",
    studentized_contrast_threshold: Number = 1,
    arrays_to_generate: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, dict, dict, int, int]:
    """
    Calculate weights of spatial associations.

    Args:
        evidential_raster: The evidential raster.
        deposits: Vector or raster data representing the mineral deposits or occurences point data.
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
        arrays_to_generate: Arrays to generate from the computed weight metrics. All column names
            in the produced weights_df are valid choices. Available column names for "unique" weights type are "Class",
            "Pixel count", "Deposit count", "W+", "S_W+", "W-", "S_W-", "Contrast", "S_Contrast", and
            "Studentized contrast". For other weights types, additional available column names are "Generalized class",
            "Generalzed W+", and "Generalized S_W+". Defaults to ["Class", "W+", "S_W+] for "unique" weights_type and
            ["Class", "W+", "S_W+", "Generalized W+", "Generalized S_W+"] for the cumulative weight types.

    Returns:
        Dataframe with weights of spatial association between the input data.
        Dictionary of arrays for specified metrics.
        Raster metadata.
        Number of deposit pixels.
        Number of all evidence pixels.

    Raises:
        ClassificationFailedException: Unable to create generalized classes for the categorical weights type.
        InvalidColumnException: Arrays to generate contains invalid column name(s).
        InvalidParameterValueException: Input weights_type is not one of the accepted values.

    Warns:
        ClassificationFailedWarning: Unable to create generalized classes for the cumulative weights types
            with the given studentized_contrast_threshold.
    """

    if arrays_to_generate is None:
        if weights_type == "unique":
            metrics_to_arrays = DEFAULT_METRICS_UNIQUE
        else:
            metrics_to_arrays = DEFAULT_METRICS_CUMULATIVE
    else:
        for col_name in arrays_to_generate:
            if col_name not in VALID_DF_COLUMNS:
                raise InvalidColumnException(f"Arrays to generate contains invalid metric / column name: {col_name}.")
        metrics_to_arrays = arrays_to_generate.copy()

    # 1. Preprocess data
    evidence_array = _read_and_preprocess_raster_data(evidential_raster, raster_nodata)
    raster_meta = evidential_raster.meta
    raster_profile = evidential_raster.profile

    # Rasterize deposits if vector data
    if isinstance(deposits, gpd.GeoDataFrame):
        deposit_array = rasterize_vector(
            geodataframe=deposits, raster_profile=raster_meta, default_value=1.0, fill_value=0.0
        )
    else:
        deposit_profile = deposits.profile

        if check_raster_grids([raster_profile, deposit_profile], same_extent=True):
            deposit_array = _read_and_preprocess_raster_data(deposits, raster_nodata)
        else:
            raise NonMatchingRasterMetadataException("Input rasters should have the same grid properties.")

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
        raise InvalidParameterValueException(
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

    # 4. If we use cumulative or categorical weights type, calculate generalized classes and weights
    if weights_type == "categorical":
        weights_df = _generalized_classes_categorical(weights_df, studentized_contrast_threshold)
        weights_df = _generalized_weights_categorical(weights_df, masked_deposit_array)
    elif weights_type == "ascending" or weights_type == "descending":
        weights_df = generalize_weights_cumulative(
            weights_df,
            classification_method="max_contrast_if_feasible",
            studentized_contrast_threshold=studentized_contrast_threshold,
        )
        if GENERALIZED_CLASS_COLUMN not in weights_df.columns.values:
            for key in GENERALIZED_COLUMNS:
                if key in metrics_to_arrays:
                    metrics_to_arrays.remove(key)

    # 5. Generate arrays for desired metrics
    arrays_dict = _generate_arrays_from_metrics(evidence_array, weights_df, metrics_to_arrays)

    # Return nr. of deposit pixels and nr. of all evidence pixels for to be used in calculate responses
    nr_of_deposits = int(np.sum(masked_deposit_array == 1))
    nr_of_pixels = int(np.size(masked_evidence_array))

    return weights_df, arrays_dict, raster_meta, nr_of_deposits, nr_of_pixels


@beartype
def weights_of_evidence_calculate_responses(
    output_arrays: Sequence[Dict[str, np.ndarray]], weights_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the posterior probabilities for the given generalized weight arrays.

    Args:
        output_arrays: List of output array dictionaries returned by weights of evidence calculations.
            For each dictionary, generalized weight and generalized standard deviation arrays are used and summed
            together pixel-wise to calculate the posterior probabilities. If generalized arrays are not found,
            the W+ and S_W+ arrays are used (so if outputs from unique weight calculations are used for this function).
        weights_df: Output dataframe of WofE calculate weights algorithm. Used for determining number of deposits and
            number of pixels.

    Returns:
        Array of posterior probabilites.
        Array of standard deviations in the posterior probability calculations.
        Array of confidence of the prospectivity values obtained in the posterior probability array.
    """
    array = list(output_arrays[0].values())[0]
    nr_of_deposits, nr_of_pixels = _calculate_nr_of_deposit_pixels(array, weights_df)

    gen_weights_sum = sum(
        [
            item[GENERALIZED_WEIGHT_PLUS_COLUMN]
            if GENERALIZED_WEIGHT_PLUS_COLUMN in item.keys()
            else item[WEIGHT_PLUS_COLUMN]
            for item in output_arrays
        ]
    )
    gen_weights_variance_sum = sum(
        [
            np.square(item[GENERALIZED_S_WEIGHT_PLUS_COLUMN])
            if GENERALIZED_S_WEIGHT_PLUS_COLUMN in item.keys()
            else np.square(item[WEIGHT_S_PLUS_COLUMN])
            for item in output_arrays
        ]
    )

    prior_probabilities = nr_of_deposits / nr_of_pixels
    prior_odds = np.log(prior_probabilities / (1 - prior_probabilities))
    posterior_probabilities = np.exp(gen_weights_sum + prior_odds) / (1 + np.exp(gen_weights_sum + prior_odds))

    posterior_probabilities_squared = np.square(posterior_probabilities)
    posterior_probabilities_std = np.sqrt(
        (1 / nr_of_deposits + gen_weights_variance_sum) * posterior_probabilities_squared
    )

    confidence_array = posterior_probabilities / posterior_probabilities_std
    return posterior_probabilities, posterior_probabilities_std, confidence_array


@beartype
def agterberg_cheng_CI_test(
    posterior_probabilities: np.ndarray, posterior_probabilities_std: np.ndarray, weights_df: pd.DataFrame
) -> Tuple[bool, bool, bool, float, str]:
    """Perform the conditional independence test presented by Agterberg-Cheng (2002).

    Agterberg, F. P. & Cheng, Q. (2002). Conditional Independence Test for Weights-of-Evidence Modeling.
    Natural Resources Research. 11. 249-255.

    Args:
        posterior_probabilities: Array of posterior probabilites.
        posterior_probabilities_std: Array of standard deviations in the posterior probability calculations.
        weights_df: Output dataframe of WofE calculate weights algorithm. Used for determining number of deposits.

    Returns:
        Whether the conditional hypothesis can be accepted for the evidence layers that the input
            posterior probabilities and standard deviations of posterior probabilities are calculated from.
        Whether the probability satisfies the 99% confidence limit.
        Whether the probability satisfies the 95% confidence limit.
        Ratio T/n. Results > 1, may be because of lack of conditional independence of layers.
            T should not exceed n by more than 15% (Bonham-Carter 1994, p. 316).
        A summary of the the conditional independence calculations.
    """
    nr_of_deposits, _ = _calculate_nr_of_deposit_pixels(posterior_probabilities, weights_df)

    # One-tailed significance test according to Agterberg-Cheng (2002):
    # Conditional independence must satisfy:
    # T - n < 1.645 * s(T) with a probability of 95%
    # T - n < 2.33 * s(T) with a probability of 99%,
    # where
    # T = the sum of posterior probabilities in all unit cells in the study area
    # n = total number of deposits

    T = np.nansum(posterior_probabilities)

    ratio = T / nr_of_deposits
    ratio_msg = "T / n > 1 may suggest lack of conditional independence.\n"
    ratio_msg_bonham_carter = "According to Bonham-Carter (1994), T / n should not exceed 1.15.\n"

    difference = T - nr_of_deposits

    T_std = np.sqrt(np.nansum(posterior_probabilities_std))

    confidence_limit_99 = 2.33 * T_std
    confidence_limit_95 = 1.645 * T_std

    confidence_99 = bool(difference < confidence_limit_99)
    confidence_95 = bool(difference < confidence_limit_95)
    sign_99 = "<" if confidence_99 else ">"
    sign_95 = "<" if confidence_95 else ">"

    conditional_independence = confidence_99 and confidence_95

    summary = f"""
    Results of conditional independence test:\n\n
    Observed number of deposits, n: {nr_of_deposits}\n
    Expected number of deposits, T: {T}\n
    Standard deviation of the expected number of deposits, s(T): {T_std}\n
    T - n = {difference}\n
    T / n = {ratio}\n{ratio_msg if ratio > 1 else ""}{ratio_msg_bonham_carter if ratio > 1.15 else ""}
    Agterberg & Cheng CI test:\n
    Data {"satisfies" if confidence_99 else "does not satisfy"} condition T - n < 2.33 * s(T):\n
    {difference} {sign_99} {confidence_limit_99}\n
    Data {"satisfies" if confidence_95 else "does not satisfy"} condition T - n < 1.645 * s(T):\n
    {difference} {sign_95} {confidence_limit_95}\n
    {"Conditional independence hypothesis should be rejected" if not conditional_independence else ""}
    """

    return conditional_independence, confidence_99, confidence_95, ratio, summary
