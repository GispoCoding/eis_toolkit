from sklearn import preprocessing
import numpy as np
from typing import Tuple


def sk_mean(a: np.array) -> Tuple[float, float, float]:
    """Tests whether it works to call one of scikit-learn's functions.

    Args:
        a: input array

    Returns:
        int: result vector containing the means of every column in the input array
    """
    scaler = preprocessing.StandardScaler().fit(a)
    ka = scaler.mean_
    return(ka)


x = np.array([
    [1., -1., 2.],
    [1., 1., 1.]
])

print(sk_mean(x))
