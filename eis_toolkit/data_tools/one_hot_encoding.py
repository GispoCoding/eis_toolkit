from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """
    One hot encode specific columns using scikit-learn.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with one-hot encoded columns.
    """

    encoder = OneHotEncoder(categories="auto", drop="first", sparse_output=False, handle_unknown="ignore", dtype=int)

    # Transform selected columns
    encoded_data = encoder.fit_transform(df)
    encoded_cols = encoder.get_feature_names_out()

    # Replace original categorical columns with encoded data
    df = pd.concat([df, pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)], axis=1)

    return df
