import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Union
from imblearn.combine import SMOTETomek

from eis_toolkit.exceptions import NonMatchingParameterLengthsException


@beartype
def balance_SMOTETomek(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    sampling_strategy: Union[float, str, dict] = "auto",
    random_state: Optional[int] = None,
) -> tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Balances the classes of input dataset using SMOTETomek resampling method.

    Args:
        X: The feature matrix (input data as a DataFrame).
        y: The target labels corresponding to the feature matrix.
        sampling_strategy: Parameter controlling how to perform the resampling.
            If float, specifies the ratio of samples in minority class to samples of majority class,
            if str, specifies classes to be resampled ("minority", "not minority", "not majority", "all", "auto"),
            if dict, the keys should be targeted classes and values the desired number of samples for the class.
            Defaults to "auto", which will resample all classes except the majority class.
        random_state: Parameter controlling randomization of the algorithm. Can be given a seed (number).
            Defaults to None, which randomizes the seed.

    Returns:
        Resampled feature matrix and target labels.

    Raises:
        NonMatchingParameterLengthsException: If X and y have different length.
    """

    if len(X) != len(y):
        raise NonMatchingParameterLengthsException("Feature matrix X and target labels y must have the same length.")

    X_res, y_res = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state).fit_resample(X, y)
    return X_res, y_res
