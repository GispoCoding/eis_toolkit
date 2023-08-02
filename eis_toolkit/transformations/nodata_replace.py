import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Union, Literal

from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal["replace", "mean", "median", "n_neighbors", "most_frequent"]
# MODE = Annotated[str,
#                  IsEqual['replace']/
#                  IsEqual['mean']/
#                  IsEqual['median']/
#                  IsEqual['n_neighbors']/
#                  IsEqual['most_frequent']
# ]
@beartype
def _nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = "replace",
    replacement_number: Optional[Union[int, float]] = 0,
    replacement_string: Optional[str] = "NaN",
    n_neighbors: Optional[int] = 2,
) -> pd.DataFrame:

    # datatype to float32 res. int32
    df.loc[:, df.dtypes == "float64"] = df.loc[:, df.dtypes == "float64"].astype("float32")
    df.loc[:, df.dtypes == "int64"] = df.loc[:, df.dtypes == "int64"].astype("int32")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # for empty String:
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    if not (len(df.columns) == 0 or len(df.index) == 0):
        if rtype == "replace":  # different between: number and string
            for cl in df.columns:
                if df[cl].dtype == "O":
                    df[cl].fillna(replacement_string, inplace=True)
                else:
                    df[cl].fillna(replacement_number, inplace=True)
        elif rtype == "n_neighbors":
            from sklearn.impute import KNNImputer

            im = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
            df = pd.DataFrame(im.fit_transform(df), columns=df.columns)
        elif rtype in ["median", "mean", "most_frequent"]:  # most_frequenty and median for categories
            from sklearn.impute import SimpleImputer

            im = SimpleImputer(missing_values=np.nan, strategy=rtype)
            df = pd.DataFrame(im.fit_transform(df), columns=df.columns)  # out: dataframe
        else:
            raise InvalidParameterValueException("nodata replacement not known: " + rtype)
    return df


# *******************************
MODE = Literal["replace", "mean", "median", "n_neighbors", "most_frequent"]
@beartype
def nodata_replace(
    df: pd.DataFrame,
    rtype: Optional[MODE] = "replace",
    replacement_number: Optional[Union[int, float]] = 0,
    replacement_string: Optional[str] = "NaN",
    n_neighbors: Optional[int] = 2,
) -> pd.DataFrame:

    """
        Replaces nodata values.
        nodata_replace.py shoud be used after separation.py (for each DataFrame separately) and befor unification.py
        There is no need to replace nan values in catagoriesed columns because nhotencoding creats a nan-class.
    Args:
        - Pandas DataFrame
        - type:
            - 'replace': Replace each nodata valu with "replacement" (see below).  Does not work for string categoriesed columns!!
            - 'medium': Replace a nodatavalue with medium of all values of the feature.
            - 'n_neighbors': Replacement calculated with k_neighbar-algorithm (see Argument n_neighbors)
            - 'most_frequent': Its's suitable for categorical columns.
        replacement_number (default = 0): Value for replacement for number columns if type is 'replace'.
        replacement_string (default = 'NaN'): Value for replacemant for string columns if type is 'replace'.
        n_neighbors (default = 2): number of neigbors if type is 'n_neighbors'

    Returns:
        - pandas dataframe without nodata values but with the same number of raws.
    """

    return _nodata_replace(
        df=df,
        rtype=rtype,
        replacement_number=replacement_number,
        replacement_string=replacement_string,
        n_neighbors=n_neighbors,
    )
