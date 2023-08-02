import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any

from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************


def _sklearn_model_fit(
    sklearnMl: Any,
    Xdf: pd.DataFrame,
    ydf: pd.DataFrame,
) -> Any:

    ty = ydf
    if len(ydf.shape) > 1:
        if ydf.shape[1] == 1:
            ty = np.ravel(ydf)

    if sklearnMl._estimator_type == "classifier":
        if np.issubdtype(ty.dtype, np.floating):
            raise InvalidParameterValueException("A classifier model cannot use a float y (target)")
    else:
        if not np.issubdtype(ty.dtype, np.number):
            raise InvalidParameterValueException("A regressor model can only use number y (target)")

    sklearnMl.fit(Xdf, ty)

    return sklearnMl


# *******************************
@beartype
def sklearn_model_fit(
    sklearnMl: Any,
    Xdf: pd.DataFrame,
    ydf: pd.DataFrame,
) -> Any:

    """
       Training of a ML model
    Args:
       - sklearnMl: before defined model (random rorest  classifier, random forest regressor, logistic regressor)
       - Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (rows)
       - ydf (Pandas dataframe or numpy array ("array-like")): target valus(columns) and samples (rows) (same number as Xdf)
          If ydf is float and the estimator is a classifier: ydf will be rounded to int.
          Returns:
         Fited ML model
    """

    # Argument evaluation
    t = (
        sklearnMl.__class__.__name__
    )
    if not t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException("DataFrame Xdf has no column")
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException("DataFrame Xdf has no rows")
    if len(ydf.columns) != 1:
        raise InvalidParameterValueException("DataFrame ydf has 0 or more then columns")
    if len(ydf.index) == 0:
        raise InvalidParameterValueException("DataFrame ydf has no rows")
    if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
        raise InvalidParameterValueException("DataFrame ydf or Xdf contains Nodata-values")

    sklearnMl = _sklearn_model_fit(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf)

    return sklearnMl
