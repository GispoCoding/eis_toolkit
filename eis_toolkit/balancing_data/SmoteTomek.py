from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek

LABELS = ["No Mineral", "Mineral"]


def load_data(path1: str, path2: str) -> pd.DataFrame:
    """
    Load two csv files from specified paths and concatenate them by rows.

    - param path1 (str): Path to the GT.
    - param path2 (str): Path to the other data
    Return:
    - dataframe: A concatenated dataframe combining rows from both CSV files
    """

    df0 = pd.read_csv(path1)  # load a dataframe from the csv file located at path1
    df1 = pd.read_csv(path2)  # load a dataframe from the csv file located at path2
    return pd.concat([df1, df0])  # concatenate and return the concatenated dataframe


def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drop specified columns from a dataframe.

    This function takes in a dataframe and a list of column names, and returns
    a new dataframe without those columns.

    Parameters:
    - df (pandas.DataFrame): The original dataframe from which columns need to be dropped.
    - columns_to_drop (list): A list of column names to drop from the dataframe.

    Returns:
    - dataframe: A new dataframe with specified columns removed.
    """

    return df.drop(columns_to_drop, axis=1)


def plot_data_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of the 'class' column from a dataframe.

    This function takes in a dataframe and visualizes the distribution
    of the 'class' column in a bar chart format. It assumes there are
    two unique values in the 'class' column and labels them using the
    global variable LABELS.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the 'class' column
                             whose distribution needs to be visualized.

    Returns:
    - None: Displays a bar chart of the class distribution.

    """

    count_classes = pd.value_counts(df["class"], sort=True)
    count_classes.plot(kind="bar", rot=0)
    plt.title("Data Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()


def data_info(df: pd.DataFrame) -> None:
    """
    Display the shape information of 'Mineral' and 'No_mineral' classes from a dataframe.

    This function takes in a dataframe and prints out the shape (number of rows and columns)
    of entries classified as 'Mineral' (class=1) and 'No_mineral' (class=0) based on the 'class' column.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the 'class' column,
                             which differentiates between 'Mineral' and 'No_mineral'.

    Returns:
    - None: Prints out the shape information for both classes.
    """

    Mineral = df[df["class"] == 1]
    No_mineral = df[df["class"] == 0]
    print(f"Mineral.shape--->{Mineral.shape}  No_mineral--->{No_mineral.shape}")


def balance_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Balance the dataset using SMOTETomek resampling method.

    This function receives features `X` and target labels `y` as inputs and balances
    the dataset using the SMOTETomek resampling technique. After resampling, it prints
    the shape of the original and the resampled dataset. It's particularly useful for
    addressing class imbalance problems.

    Parameters:
    - X (pandas.DataFrame or pandas.Series): The feature matrix.
    - y (pandas.Series): The target labels corresponding to the feature matrix.

    Returns:
    - tuple: Resampled feature matrix and target labels (X_res, y_res).

    """

    smk = SMOTETomek(random_state=42)
    X_res, y_res = smk.fit_resample(X, y)
    print("Original dataset shape {}".format(Counter(y)))
    print("Resampled dataset shape {}".format(Counter(y_res)))
    return X_res, y_res
