import numpy as np
import pandas as pd
import rasterio
from typing import Literal, Tuple, List, Union, Optional
from numbers import Number
from beartype import beartype
from functools import partial


SMALL_NUMBER = 0.0001
LARGE_NUMBER = 1.0001
# NODATA_THRESHOLD = 0.0000001


def read_and_preprocess_raster(raster: rasterio.io.DatasetReader, nodata: Optional[Number]) -> np.ndarray:
    """Read raster data and handle NoData values."""
    array = np.array(raster.read(1), dtype=np.float32)

    if nodata is not None:
        nan_mask = np.isclose(array, np.full(raster.shape, nodata))
        array[nan_mask] = np.nan
    elif raster.meta["nodata"] is not None:
        array[array == raster.meta["nodata"]] = np.nan

    return array


def calculate_metrics_for_class(deposits: np.ndarray, evidence: np.ndarray):
    A = np.sum(np.logical_and(deposits == 1, evidence == 1))  # Deposit and evidence present
    B = np.sum(np.logical_and(deposits == 1, evidence == 0))  # Depsoti present and evidence absent
    C = np.sum(np.logical_and(deposits == 0, evidence == 1))  # Deposit absent and evidence present
    D = np.sum(np.logical_and(deposits == 0, evidence == 0))  # Depsoti and evidence absent

    CONSTANT = 0.5
    LAPLACE_SMOOTHING = False
    REPLACE = True

    if A + B == 0:
        raise Exception("No deposits")
    if C + D == 0:
        raise Exception("No evidence")

    if LAPLACE_SMOOTHING:
        p_A = (A + CONSTANT) / (A + B + 2*CONSTANT)
        p_C = (C + CONSTANT) / (C + D + 2*CONSTANT)
    elif REPLACE:
        # The 4 lines below are not needed to avoid errors, but are needed to replicate the original implementation
        if A == 0:
            A = SMALL_NUMBER
        if C == 1:
            C = LARGE_NUMBER
        p_A = A / (A + B)
        p_C = C / (C + D)
        if p_A == 0:
            p_A = SMALL_NUMBER
        elif p_A == 1:
            p_A = LARGE_NUMBER
        if p_C == 0:
            p_C = SMALL_NUMBER
        elif p_C == 1:
            p_C = LARGE_NUMBER
    else:
        p_A = A / (A + B) # probability of presence of evidence given the presence of mineral deposit
        p_C = C / (C + D) # probability of presence of evidence given the absence of mineral deposit

    w_plus = np.log(p_A / p_C) if p_A != 0 and p_C != 0 else 0
    w_minus = np.log((1 - p_A) / (1 - p_C)) if (1 - p_A) != 0 and (1 - p_C) != 0 else 0
    contrast = w_plus - w_minus

    s_w_plus = np.sqrt((1 / A if A != 0 else 0) + (1 / C if C != 0 else 0))
    s_w_minus = np.sqrt((1 / B if B != 0 else 0) + (1 / D if D != 0 else 0))
    s_contrast = np.sqrt(s_w_plus**2 + s_w_minus**2)

    return A, B, C, D, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast


def unique_weights(event: np.ndarray, condition: np.ndarray) -> dict:
    classes = np.unique(condition[~np.isnan(condition)])
    return {cls: calculate_metrics_for_class(event, condition == cls) for cls in classes}


def cumulative_weights(event: np.ndarray, condition: np.ndarray, ascending: bool = True) -> dict:
    classes = sorted(np.unique(condition[~np.isnan(condition)]), reverse=not ascending)
    cumulative_classes = [classes[:i+1] for i in range(len(classes))]
    return {tuple(cls): calculate_metrics_for_class(event, np.isin(condition, cls)) for cls in cumulative_classes}


def reclass_by_studentized_contrast(df: pd.DataFrame, studentized_contrast_threshold: float):
    """Reclassifies based on the studentized contrast value."""    
    df['Reclassified'] = np.where(df['Studentized contrast'] >= studentized_contrast_threshold, 2, 1)

    # Check if both classes are present
    unique_classes = df['Reclassified'].unique()
    if 1 not in unique_classes:
        raise ValueError("Reclassification failed: 'Unfavorable' class (Class 1) doesn't exist.")
    elif 2 not in unique_classes:
        raise ValueError("Reclassification failed: 'Favorable' class (Class 2) doesn't exist.")


# def generate_raster_for_metric(condition: np.ndarray, classes: Union[int, Tuple[int, ...]], metric_value: float) -> np.ndarray:
#     """
#     Generates a raster where cells of specified classes are replaced with a metric value.
#     Other cells are set to NaN.
#     """
#     raster = np.full(condition.shape, np.nan)
#     mask = np.isin(condition, classes)
#     raster[mask] = metric_value
#     return raster


# def generate_rasters_from_metrics(evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str] = ["Class", "WPlus", "S_WPlus"]) -> dict:
#     """
#     Generates rasters for defined metrics based on the Weights of Evidence calculations.
#     """
#     raster_dict = {}
#     for metric in metrics_to_include:
#         raster = np.full(evidence.shape, np.nan)
#         for cls in df["Class"]:
#             mask = np.isin(evidence, cls)
#             raster[mask] = df["Class"][metric]
#     return raster_dict


def generate_rasters_from_metrics(evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str] = ["Class", "WPlus", "S_WPlus"]) -> dict:
    """
    Generates rasters for defined metrics based on the Weights of Evidence calculations.
    """
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
    deposit_raster: rasterio.io.DatasetReader,
    weights_type: Literal['unique', 'ascending', 'descending'] = 'unique',
    studentized_contrast: float = 2,
) -> Tuple[pd.DataFrame, dict, dict]:

    # 1. Data preprocessing
    deposits = read_and_preprocess_raster(deposit_raster)
    evidence = read_and_preprocess_raster(evidential_raster)

    # 2. WoE Calculation
    if weights_type == 'unique':
        woe_weights = unique_weights(deposits, evidence)
    elif weights_type == 'ascending':
        woe_weights = cumulative_weights(deposits, evidence, ascending=True)
    elif weights_type == 'descending':
        woe_weights = cumulative_weights(deposits, evidence, ascending=False)

    # Calculate additional columns based on adjusted_weights
    df_entries = []
    for classes, metrics in woe_weights.items():
        metrics = [round(metric, 4) for metric in metrics]
        A, _, C, _, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast = metrics

        df_entries.append({
            'Class': classes,
            'Count': A + C,
            'Point Count': A,
            'WPlus': w_plus,
            'S_WPlus': s_w_plus,
            'WMinus': w_minus,
            'S_WMinus': s_w_minus,
            'Contrast': contrast,
            'S_Contrast': s_contrast,
            'Studentized contrast': contrast / s_contrast
        })

    # 4. Create DataFrame
    weights_df = pd.DataFrame(df_entries)

    if weights_type != 'unique':
        reclass_by_studentized_contrast(weights_df, studentized_contrast)

    # After the woe_weights computation in the weights_of_evidence function
    raster_dict = generate_rasters_from_metrics(evidence, weights_df, ["Class", "WPlus", "S_WPlus"])

    # 6. Extract raster metadata
    raster_meta = evidential_raster.meta

    return weights_df, raster_dict, raster_meta