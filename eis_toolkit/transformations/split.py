import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from sklearn.model_selection import train_test_split

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _split(
    Xdf: pd.DataFrame,  # dataframe of Features for traning (to split in training and test dataset) and or for Xdf abnd ydf
    ydf: Optional[
        pd.DataFrame
    ] = None,  # dataframe of known values for training (to split) or known values to compare with test_y
    test_size: Optional[Union[int, float]] = None,  # int: number of test-samples, if float: 0<ts<1
    train_size: Optional[Union[int, float]] = None,  # if None: complement size of the test_size
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:

    # check:
    if test_size is not None:
        if test_size <= 0:
            test_size = None
    if train_size is not None:
        if train_size <= 0:
            train_size = None
    if test_size is None and train_size is None:
        test_size = 0.25

    # split in test and training datasets
    if ydf is not None:
        train_Xdf, test_Xdf, train_ydf, test_ydf = train_test_split(
            Xdf, ydf, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle
        )
    else:
        train_ydf = test_ydf = None
        train_Xdf, test_Xdf = train_test_split(
            Xdf, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle
        )

    return train_Xdf, test_Xdf, train_ydf, test_ydf


# *******************************
@beartype
def split(
    Xdf: pd.DataFrame,  # dataframe of Features for splitting in traning and test dataset
    ydf: Optional[pd.DataFrame] = None,  # dataframe of known values for splitting
    test_size: Optional[Union[int, float]] = None,  # int: namuber of test-samples, if float: 0<ts<1
    train_size: Optional[Union[int, float]] = None,  # if None: complement size of the test_size
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:

    """
        Splits the rows of Xdf and ydf (if given) in Training and Test-Set
        - random splited testset from size test_size/train_size.
          Xdf and ydf will be randomly splitted in a test and a training dataset.
        The funcion may de used to create a validation dataset before starting the training process

    Args:
        - Xdf ("array-like") features (columns) and samples (rows)
        - ydf (optional, "array-like") target valus(column) and samples (rows) (same number as Xdf)
        - test_size:
           If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
           If int, represents the absolute number of test samples.
           If None or negative, the value is set to the complement of the train size.
             If train_size is also None or negative, it will be set to 0.25.
        - train_size:
           If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
           If int, represents the absolute number of train samples.
           If None or negative, the value is automatically set to the complement of the test size.
        - random_state:
           Controls the shuffling applied to the data before applying the split.
           Pass an int for reproducible output across multiple function calls.
           No effect with selftest.
        - shuffle:
           Whether or not to shuffle the data before splitting.
           No effect with selftest
     Returns:
        following 4 dataframes:
        - X for training (training set)
        - X for test (test set)
        - y (optional) for training (know target for training)
        - y (optional) for test (known target for test)

    """

    # Argument evaluation
    if Xdf is not None:
        if len(Xdf.columns) == 0 or len(Xdf.index) == 0:
            raise InvalidParameterValueException("DataFrame has no column or no rows")
        if ydf is not None:
            if len(Xdf) != len(ydf):
                raise InvalidParameterValueException("DataFrames have different numbers of rows")

    train_Xdf, test_Xdf, train_ydf, test_ydf = _split(
        Xdf=Xdf,
        ydf=ydf,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return train_Xdf, test_Xdf, train_ydf, test_ydf
