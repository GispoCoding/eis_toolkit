import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any

from eis_toolkit.exceptions import InvalidParameterValueException  # , InvalideContentOfInputDataFrame


# *********************************
@beartype
def _sklearn_check_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of features for prediction
) -> pd.DataFrame:

    if not (Xdf.shape[1] == Xdf.select_dtypes(include=np.number).shape[1]):
        raise InvalidParameterValueException("Non numeric data in the Dataframe")
    else:
        Xdf = Xdf.reindex(columns=sklearnMl.feature_names_in_)  # alternative

    return Xdf


# *******************************
@beartype
def sklearn_check_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of features for prediction
) -> pd.DataFrame:
    """
        Check the fields of input DataFrame.

        Check whether the fields of the dataframe Xdf are the same (amount and names)
        as the fields in the model sklearnMl.
        More columns will be dropped.
        The Columns will be ordered in the same way as in sklearnMl used.
        Check_Prediction should be used just befor model_prediction and after onehotencoding.

    Args:
        - sklearnMl: Existing model to use for the prediction
                   (random forest classifier, random forest regressor, logistic regression)
        - Xdf ("array-like"): features (columns) and samples (raws)

    Returns:
        Dataframe with columns in the same order of the model sklearnMl
    """
    t = (
        sklearnMl.__class__.__name__
    )  # t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
    if t not in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "Argument sklearnMl is not RandomForestClassifier, RandomForestRegressor or LogisticRegression)"
        )
    # Fields in sklearnMl are in Xdf as well? (Reduce Xdf-columns to the list out of the model)
    t0 = set(Xdf.columns) - set(sklearnMl.feature_names_in_)
    tdf = Xdf.drop(columns=(t0))
    t0 = set(sklearnMl.feature_names_in_) - set(tdf.columns)
    if t0.__len__() > 0:
        raise InvalidParameterValueException(
            "Missing columns in dataframe compared with Model: "
            + str(t0)
        )
    # t0 = set(Xdf.columns) - set(sklearnMl.feature_names_in_)
    # if t0.__len__() > 0:
    #     raise InvalidParameterValueException("More columns in dataframe (compared with Model): " + str(t0))

    return _sklearn_check_prediction(sklearnMl, tdf)
