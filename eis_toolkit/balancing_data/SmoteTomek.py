from beartype import beartype
from beartype.typing import Tuple
from eis_toolkit.exceptions import EmptyDataFrameException
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek

LABELS = ["No Mineral", "Mineral"]


@beartype
def smote_tomek(features: pd.DataFrame, target_label: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance the dataset using SMOTETomek resampling method.
    This function receives features `X` and target labels `y` as inputs and balance 
    the dataset using the SMOTETomek resampling technique. After resampling, it prints
    the shape of the original and the resampled dataset. It's particularly useful for
    addressing class imbalance problems.

    Parameters:
        X (pandas.DataFrame or pandas.Series): The feature matrix.
        y (pandas.Series): The target labels corresponding to the feature matrix.

    Returns:
        tuple: Resampled feature matrix and target labels (X_res, y_res).

 
    """
    if features is None or features.empty:
        raise EmptyDataFrameException
    smk = SMOTETomek(random_state=42)
    X_res, y_res = smk.fit_resample(features, target_label)
    return X_res, y_res
