import numpy as np
import sklearn.preprocessing


def normalize_the_data(scaler_agent: sklearn.preprocessing, data: np.ndarray):
    """
    Normalize multidimensional data using a specified scaler.

    This function applies a scaling technique to each feature in the data, transforming the data to a specified
    range or distribution. It is particularly useful for normalizing image data or similarly structured
    multidimensional data.
    Parameters:
        - scaler_agent: An instance of a preprocessing scaler from sklearn.preprocessing (e.g., MinMaxScaler,
        StandardScaler) that will be used to apply normalization.
        - data: A NumPy ndarray containing the data to be normalized. The data is expected to be in a multidimensional
        format, typically including dimensions for sample size, height, width, and channels (e.g., for image data).
    Returns:
        - normalized_data: The input data normalized according to the scaler_agent. The output data maintains
        the original shape of the input data.

    """
    number_of_sample, h, w, c = data.shape
    temp = scaler_agent.transform(data.reshape(-1, data.shape[-1]))
    normalized_data = temp.reshape(number_of_sample, h, w, c)
    return normalized_data
