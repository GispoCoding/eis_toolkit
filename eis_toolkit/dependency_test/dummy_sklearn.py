from sklearn import preprocessing
import numpy as np


def sk_mean(a: np.array) -> np.array:
    """Tests whether it works to call one of scikit-learn's functions.

    Args:
        a (np.array): input array

    Returns:
        np.array: result vector containing the means of every column in the
        input array
    """
    scaler = preprocessing.StandardScaler().fit(a)
    ka = scaler.mean_
    return(ka)


x = np.array([
    [1., -1., 2.],
    [1., 1., 1.]
])

print(sk_mean(x))
