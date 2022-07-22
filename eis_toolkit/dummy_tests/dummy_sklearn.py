from sklearn import preprocessing
import numpy as np


def sk_mean(a: np.ndarray) -> np.ndarray:
    """Tests whether it works to call one of scikit-learn's functions.

    Args:
        a (np.ndarray): input array

    Returns:
        np.ndarray: result vector containing the means of every column in the
        input array
    """
    scaler = preprocessing.StandardScaler().fit(a)
    ka = scaler.mean_
    return(ka)
