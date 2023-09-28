from numbers import Number

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import List, Literal, Optional, Sequence, Tuple, Union

from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector


def read_and_preprocess_evidence(raster: rasterio.io.DatasetReader, nodata: Optional[Number] = None) -> np.ndarray:
    """Read raster data and handle NoData values."""

    array = np.array(raster.read(1), dtype=np.float32)

    if nodata is not None:
        array[array == nodata] = np.nan
    elif raster.meta["nodata"] is not None:
        array[array == raster.meta["nodata"]] = np.nan

    return array


def calculate_metrics_for_class(deposits: np.ndarray, evidence: np.ndarray) -> tuple:
    """Calculate weights/metrics for given data."""
    A = np.sum(np.logical_and(deposits == 1, evidence == 1))
    B = np.sum(np.logical_and(deposits == 1, evidence == 0))
    C = np.sum(np.logical_and(deposits == 0, evidence == 1))
    D = np.sum(np.logical_and(deposits == 0, evidence == 0))

    if A + B == 0:
        raise Exception("No deposits")
    if C + D == 0:
        raise Exception("All included cells have deposits")

    if A == 0:
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


def unique_weights(deposits: np.ndarray, evidence: np.ndarray) -> dict:
    """Calculate unique weights for each class."""
    classes = np.unique(evidence)
    return {cls: calculate_metrics_for_class(deposits, evidence == cls) for cls in classes}


def cumulative_weights(deposits: np.ndarray, evidence: np.ndarray, ascending: bool = True) -> dict:
    """Calculate cumulative weights (ascending or descending) for each class."""
    classes = sorted(np.unique(evidence), reverse=not ascending)
    cumulative_classes = [classes[: i + 1] for i in range(len(classes))]
    return {
        cls[i]: calculate_metrics_for_class(deposits, np.isin(evidence, cls))
        for i, cls in enumerate(cumulative_classes)
    }


def reclassify_by_studentized_contrast(df: pd.DataFrame, studentized_contrast_threshold: Number) -> None:
    """Create generalized classes based on the studentized contrast threhsold value."""
    index = df.idxmax()["Contrast"]

    if df.loc[index, "Studentized contrast"] < studentized_contrast_threshold:
        raise Exception("Failed, studentized contrast is {}".format(df.loc[index, "Studentized contrast"]))

    df["Generalized class"] = 1
    for i in range(0, index + 1):
        df.loc[i, "Generalized class"] = 2


def reclassify_by_studentized_contrast_alternative(df: pd.DataFrame, studentized_contrast_threshold: Number) -> None:
    """Create generalized classes based on the studentized contrast threhsold value."""
    df["Generalized class"] = np.where(df["Studentized contrast"] >= studentized_contrast_threshold, 2, 1)

    # Check if both classes are present
    unique_classes = df["Generalized class"].unique()
    if 1 not in unique_classes:
        raise ValueError("Reclassification failed: 'Unfavorable' class (Class 1) doesn't exist.")
    elif 2 not in unique_classes:
        raise ValueError("Reclassification failed: 'Favorable' class (Class 2) doesn't exist.")


def calculate_generalized_weights(df: pd.DataFrame, deposits) -> None:
    """
    Calculate generalized weights.

    Implementation for generalized weights that uses the SAME logic than the original implementation.
    """
    total_deposits = np.sum(deposits == 1)
    total_no_deposits = deposits.size - total_deposits

    # Class 2
    class_2_max_index = df.idxmax()["Contrast"]
    class_2_count = df.loc[class_2_max_index, "Pixel count"]
    class_2_point_count = df.loc[class_2_max_index, "Deposit count"]

    class_2_w_gen = np.log(class_2_point_count / total_deposits) - np.log(
        (class_2_count - class_2_point_count) / total_no_deposits
    )
    clas_2_s_wpls_gen = np.sqrt((1 / class_2_point_count) + (1 / (class_2_count - class_2_point_count)))

    df["Generalized W+"] = round(class_2_w_gen, 4)
    df["Generalized S_W+"] = round(clas_2_s_wpls_gen, 4)

    # Class 1
    class_1_count = df.loc[len(df.index) - 1, "Pixel count"] - class_2_count
    class_1_point_count = df.loc[len(df.index) - 1, "Deposit count"] - class_2_point_count

    class_1_w_gen = np.log(class_1_point_count / total_deposits) - np.log(
        (class_1_count - class_1_point_count) / total_no_deposits
    )
    clas_1_s_wpls_gen = np.sqrt((1 / class_1_point_count) + (1 / (class_1_count - class_1_point_count)))
    df.loc[df["Generalized class"] == 1, "Generalized W+"] = round(class_1_w_gen, 4)
    df.loc[df["Generalized class"] == 1, "Generalized S_W+"] = round(clas_1_s_wpls_gen, 4)


def calculate_generalized_weights_alternative(weights_df: pd.DataFrame) -> None:
    """
    Calculate generalized weights.

    Implementation for generalized weights that uses a DIFFERENT logic than the original implementation.
    """
    generalized_weights = []
    generalized_s_weights = []

    for gen_cls in weights_df["Generalized class"].tolist():
        subset_df = weights_df[weights_df["Generalized class"] == gen_cls]

        weighted_w_plus_sum = sum(subset_df["WPlus"] * subset_df["Pixel count"])
        total_count = subset_df["Deposit count"].sum()

        generalized_weights.append(round(weighted_w_plus_sum / total_count, 4) if total_count else 0)

    weights_df["Generalized W+"] = generalized_weights
    weights_df["Generalized S_W+"] = generalized_s_weights


def generate_rasters_from_metrics(
    evidence: np.ndarray, df: pd.DataFrame, metrics_to_include: List[str] = ["Class", "W+", "S_W+"]
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


@beartype
def weights_of_evidence(
    evidential_raster: rasterio.io.DatasetReader,
    deposits: gpd.GeoDataFrame,
    raster_nodata: Optional[Number] = None,
    weights_type: Literal["unique", "ascending", "descending"] = "unique",
    studentized_contrast_threshold: Number = 2,
    rasters_to_generate: Union[Sequence[str], str, None] = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Calculate weights of spatial associations.

    Args:
        evidential_raster: The evidential raster.
        deposits: Vector data representing the mineral deposits or occurences point data.
        raster_nodata: If nodata value of raster is wanted to specify manually. Optional parameter, defaults to None
            (nodata from raster metadata is used).
        weights_type: Accepted values are 'unique' for unique weights, 'ascending' for cumulative ascending weights,
            'descending' for cumulative descending weights. Defaults to 'unique'.
        studentized_contrast_threshold: Studentized contrast threshold value used to reclassify all classes.
            Reclassification is used when creating generalized rasters with cumulative weight type selection.
            Not needed if weights_type is 'unique'. Defaults to 2.
        rasters_to_generate: Rasters to generate from the computed weight metrics. All column names
            in the produced weights_df are valid choices. If None, defaults to ["Class", "W+", "S_W+]
            for "unique" weights_type or ["Class", "W+", "S_W+", "Generalized W+", "Generalized S_W+"]
            for the cumulative weight types.

    Returns:
        Dataframe with weights of spatial association between the input rasters.
        Dictionary of output raster arrays.
        Raster metadata.
    """

    # 1. Data preprocessing

    # Read evidence raster
    evidence_array = read_and_preprocess_evidence(evidential_raster, raster_nodata)

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
        wofe_weights = unique_weights(masked_deposit_array, masked_evidence_array)
    elif weights_type == "ascending":
        wofe_weights = cumulative_weights(masked_deposit_array, masked_evidence_array, ascending=True)
    elif weights_type == "descending":
        wofe_weights = cumulative_weights(masked_deposit_array, masked_evidence_array, ascending=False)

    # 3. Create dataframe based on calculated metrics
    df_entries = []
    for cls, metrics in wofe_weights.items():
        metrics = [round(metric, 4) if isinstance(metric, np.floating) else metric for metric in metrics]
        A, _, C, _, w_plus, s_w_plus, w_minus, s_w_minus, contrast, s_contrast, studentized_contrast = metrics
        df_entries.append(
            {
                "Class": cls,
                "Pixel count": A + C,
                "Deposit count": A,
                "W+": w_plus,
                "S_W+": s_w_plus,
                "W-": w_minus,
                "S_W-": s_w_minus,
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
        calculate_generalized_weights(weights_df, masked_deposit_array)

    metrics_to_rasters = rasters_to_generate
    if metrics_to_rasters is None:
        metrics_to_rasters = ["Class", "W+", "S_W+"]
        if weights_type != "unique":
            metrics_to_rasters += ["Generalized W+", "Generalized S_W+"]

    # 5. After the wofe_weights computation in the weights_of_evidence function
    raster_dict = generate_rasters_from_metrics(evidence_array, weights_df, metrics_to_rasters)

    return weights_df, raster_dict, raster_meta
