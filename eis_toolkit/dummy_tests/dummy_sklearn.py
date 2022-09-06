from typing import Any

import numpy as np
from sklearn import preprocessing


def sk_mean(
    a: np.ndarray,
) -> Any:
    """Test whether it works to call one of scikit-learn's functions.

    Args:
        a (np.ndarray): input array.

    Returns:
        np.ndarray: result vector containing the means of every column in the input array.
    """
    scaler = preprocessing.StandardScaler().fit(a)
    ka = scaler.mean_

    return ka
