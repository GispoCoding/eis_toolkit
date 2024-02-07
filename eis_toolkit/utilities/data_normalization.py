import numpy as np
import sklearn.preprocessing
from beartype import beartype


@beartype
def normalize_the_data(scaler_agent: sklearn.preprocessing, data: np.ndarray):
    """
    Do Data normalization.

    Parameters:
       scaler_agent: this is the scaler agent used for data normalization is like an handler.
       data: data to normalize

    Return:
        return normalized data.
    """
    number_of_sample, h, w, c = data.shape
    temp = scaler_agent.transform(data.reshape(-1, data.shape[-1]))
    normalized_data = temp.reshape(number_of_sample, h, w, c)
    return normalized_data
