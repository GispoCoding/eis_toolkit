from numbers import Number
from typing import List, Literal, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

# from beartype import beartype

# REPLACE signifies if we use replacement of 0 and 1 with the values below
# If REPLACE is False but laplace_smoothing is set to True, we use the SMOOTH_CONSTANT to compute p_A and p_C
REPLACE = False
SMALL_NUMBER = 0.0001
LARGE_NUMBER = 1.0001

SMOOTH_CONSTANT = 1.0

# NODATA_THRESHOLD = 0.0000001


def read_and_preprocess_evidence(raster: rasterio.io.DatasetReader, nodata: Optional[Number] = None) -> np.ndarray:
    """Read raster data and handle NoData values."""
    array = np.array(raster.read(1), dtype=np.float32)

    if nodata is not None:
        array[array == nodata] = np.nan
    elif raster.meta["nodata"] is not None:
        array[array == raster.meta["nodata"]] = np.nan

    return array


def calculate_metrics_for_class(deposits: np.ndarray, evidence: np.ndarray, laplace_smoothing: bool):
    """Calculate weights/metrics for given data."""
    A = np.sum(np.logical_and(deposits == 1, evidence == 1))
    B = np.sum(np.logical_and(deposits == 1, evidence == 0))
    C = np.sum(np.logical_and(deposits == 0, evidence == 1))
    D = np.sum(np.logical_and(deposits == 0, evidence == 0))

    if A + B == 0:
        raise Exception("No deposits")
    if C + D == 0:
        raise Exception("All included cells have deposits")

    if not laplace_smoothing:
        p_A = A / (A + B)  # probability of presence of evidence given the presence of mineral deposit
        p_C = C / (C + D)  # probability of presence of evidence given the absence of mineral deposit
    else:
        p_A = (A + SMOOTH_CONSTANT) / (A + B + 2 * SMOOTH_CONSTANT)
        p_C = (C + SMOOTH_CONSTANT) / (C + D + 2 * SMOOTH_CONSTANT)

    # Calculate metrics
    w_plus = np.log(p_A / p_C) if p_A != 0 and p_C != 0 else 0
    w_minus = np.log((1 - p_A) / (1 - p_C)) if (1 - p_A) != 0 and (1 - p_C) != 0 else 0
    contrast = w_plus - w_minus

    # Calculate signifigance metrics
    s_w_plus = np.sqrt((1 / A if A != 0 else 0) + (1 / C if C != 0 else 0))
    s_w_minus = np.sqrt((1 / B if B != 0 else 0) + (1 / D if D != 0 else 0))
    s_contrast = np.sqrt(s_w_plus**2 + s_w_minus**2)

    # Calculate studentized contrast
    studentized_contrast = contrast / s_contrast

    return A, B, C, D, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast, studentized_contrast


def unique_weights(deposits: np.ndarray, evidence: np.ndarray, laplace_smoothing: bool) -> dict:
    """Calculate unique weights for each class."""
    classes = np.unique(evidence)
    return {cls: calculate_metrics_for_class(deposits, evidence == cls, laplace_smoothing) for cls in classes}


def cumulative_weights(
    deposits: np.ndarray, evidence: np.ndarray, laplace_smoothing: bool, ascending: bool = True
) -> dict:
    """Calculate cumulative weights (ascending or descending) for each class."""
    classes = sorted(np.unique(evidence), reverse=not ascending)
    cumulative_classes = [classes[: i + 1] for i in range(len(classes))]
    return {
        cls[i]: calculate_metrics_for_class(deposits, np.isin(evidence, cls), laplace_smoothing)
        for i, cls in enumerate(cumulative_classes)
    }


def reclassify_by_studentized_contrast(df: pd.DataFrame, studentized_contrast_threshold: Number) -> None:
    """Create generalized classes based on the studentized contrast threhsold value."""
    df["Generalized class"] = np.where(df["Studentized contrast"] >= studentized_contrast_threshold, 2, 1)

    # Check if both classes are present
    unique_classes = df["Generalized class"].unique()
    if 1 not in unique_classes:
        raise ValueError("Reclassification failed: 'Unfavorable' class (Class 1) doesn't exist.")
    elif 2 not in unique_classes:
        raise ValueError("Reclassification failed: 'Favorable' class (Class 2) doesn't exist.")


def calculate_generalized_weights(weights_df: pd.DataFrame) -> None:
    """
    Calculate generalized weights.

    Implementation for generalized weights that uses a DIFFERENT logic than the original implementation.
    """
    generalized_weights = []
    generalized_s_weights = []

    for gen_cls in weights_df["Generalized class"].tolist():
        subset_df = weights_df[weights_df["Generalized class"] == gen_cls]

        weighted_w_plus_sum = sum(subset_df["WPlus"] * subset_df["Count"])
        total_count = subset_df["Count"].sum()

        generalized_weights.append(round(weighted_w_plus_sum / total_count, 4) if total_count else 0)

    weights_df["Generalized WPlus"] = generalized_weights
    weights_df["Generalized S_WPlus"] = generalized_s_weights


def calculate_generalized_weights_alternative(weights_df: pd.DataFrame, deposits) -> None:
    """
    Calculate generalized weights.

    Implementation for generalized weights that uses the SAME logic as the original implementation.
    """
    total_deposits = np.sum(deposits == 1)
    total_no_deposits = deposits.size - total_deposits

    generalized_weights = []
    generalized_s_weights = []

    for gen_cls in weights_df["Generalized class"].tolist():
        subset_df = weights_df[weights_df["Generalized class"] == gen_cls]

        cumulative_deposit_count = subset_df["Point Count"].sum()
        cumulative_no_deposit_count = subset_df["Count"].sum() - cumulative_deposit_count

        W_Gen = np.log(cumulative_deposit_count / total_deposits) - np.log(
            cumulative_no_deposit_count / total_no_deposits
        )
        s_wpls_gen = np.sqrt((1 / cumulative_deposit_count) + (1 / cumulative_no_deposit_count))

        generalized_weights.append(round(W_Gen, 4))
        generalized_s_weights.append(round(s_wpls_gen, 4))

    weights_df["Generalized WPlus"] = generalized_weights
    weights_df["Generalized S_WPlus"] = generalized_s_weights


def generate_rasters_from_metrics(
    evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str] = ["Class", "WPlus", "S_WPlus"]
) -> dict:
    """Generate rasters for defined metrics based."""
    raster_dict = {}
    for metric in metrics_to_include:
        raster = np.full(evidence.shape, np.nan)
        for _, row in df.iterrows():
            mask = np.isin(evidence, row["Class"])
            raster[mask] = row[metric]
        raster_dict[metric] = raster
    return raster_dict


# @beartype
def weights_of_evidence(
    evidential_raster: rasterio.io.DatasetReader,
    deposits: gpd.GeoDataFrame,
    weights_type: Literal["unique", "ascending", "descending"] = "unique",
    studentized_contrast_threshold: Number = 2,
    laplace_smoothing: bool = False,
    rasters_to_generate: Union[Sequence[str], str, None] = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Calculate weights of spatial associations.

    Args:
        evidential_raster: The evidential raster.
        deposits: Vector data representing the mineral deposits or occurences point data.
        weights_type: Accepted values are 'unique' for unique weights, 'ascending' for cumulative ascending weights,
            'descending' for cumulative descending weights. Defaults to 'unique'.
        studentized_contrast_threshold: Studentized contrast threshold value used to reclassify all classes.
            Reclassification is used when creating generalized rasters with cumulative weight type selection.
            Not needed if weights_type is 'unique'. Defaults to 2.
        laplace_smoothing: If smoothing is applied in logarithmic calculations. If no smoothing is applied,
            the problematic cases result into weight value of 0 for the class. Defaults to False.
        rasters_to_generate: Rasters to generate from the computed weight metrics. All column names
            in the produced weights_df are valid choices. If None, defaults to ["Class", "WPlus", "S_WPlus"]
            for "unique" weights_type or ["Class", "WPlus", "S_WPlus", "Generalized WPlus", "Generalized S_WPlus"]
            for the cumulative weight types.

    Returns:
        Dataframe with weights of spatial association between the input rasters.
        Dictionary of output raster arrays.
        Raster metadata.
    """

    # 1. Data preprocessing

    # Read evidence raster
    evidence_array = read_and_preprocess_evidence(evidential_raster)

    # Extract raster metadata
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
    if weights_type == "unique":
        wofe_weights = unique_weights(masked_deposit_array, masked_evidence_array, laplace_smoothing)
    elif weights_type == "ascending":
        wofe_weights = cumulative_weights(
            masked_deposit_array, masked_evidence_array, laplace_smoothing, ascending=True
        )
    elif weights_type == "descending":
        wofe_weights = cumulative_weights(
            masked_deposit_array, masked_evidence_array, laplace_smoothing, ascending=False
        )

    # 3. Create dataframe based on calculated metrics
    df_entries = []
    for cls, metrics in wofe_weights.items():
        metrics = [round(metric, 3) if isinstance(metric, np.floating) else metric for metric in metrics]
        A, _, C, _, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast, studentized_contrast = metrics
        df_entries.append(
            {
                "Class": cls,
                "Count": A + C,
                "Point Count": A,
                "WPlus": w_plus,
                "S_WPlus": s_w_plus,
                "WMinus": w_minus,
                "S_WMinus": s_w_minus,
                "Contrast": contrast,
                "S_Contrast": s_contrast,
                "Studentized contrast": studentized_contrast,
            }
        )
    weights_df = pd.DataFrame(df_entries)

    # 4. If we use cumulative weights type, reclassify and calculate generalized weights
    if weights_type != "unique":
        reclassify_by_studentized_contrast(weights_df, studentized_contrast_threshold)
        # calculate_generalized_weights(weights_df)
        calculate_generalized_weights_alternative(weights_df, masked_deposit_array)

    metrics_to_rasters = rasters_to_generate
    if metrics_to_rasters is None:
        metrics_to_rasters = ["Class", "WPlus", "S_WPlus"]
        if weights_type != "unique":
            metrics_to_rasters += ["Generalized WPlus", "Generalized S_WPlus"]

    # 5. After the wofe_weights computation in the weights_of_evidence function
    raster_dict = generate_rasters_from_metrics(evidence_array, weights_df, metrics_to_rasters)

    return weights_df, raster_dict, raster_meta
