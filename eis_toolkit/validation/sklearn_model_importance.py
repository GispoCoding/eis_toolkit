import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Union
from sklearn.inspection import permutation_importance

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _sklearn_model_importance(
    sklearnMl: Any,  # should be fitted model
    Xdf: Optional[pd.DataFrame] = None,  # dataframe for permutation importance
    ydf: Optional[pd.DataFrame] = None,
    n_repeats: Optional[int] = 5,  # number of permutation, default = 5
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    max_samples: Optional[Union[float, int]] = 1.0,
) -> pd.DataFrame:

    importance = None
    fields = sklearnMl.feature_names_in_
    if (
        sklearnMl.__str__().find("RandomForestClassifier") >= 0
        or sklearnMl.__str__().find("RandomForestRegressor") >= 0
    ):
        trf = sklearnMl.feature_importances_
        importance = pd.DataFrame(zip(fields, trf))  # for pd.DataFrame  dict is possible as well
    else:
        if Xdf is None or ydf is None:
            raise InvalidParameterValueException(
                "estimator is not RandomForest, Xdf and ydf must be given"
            )  # feature importance from Random Forest
    if Xdf is not None and ydf is not None:  # Permutation feature importance
        if len(Xdf.columns) == 0:
            raise InvalidParameterValueException.append("DataFrame Xdf has no column")
        if len(Xdf.index) == 0:
            raise InvalidParameterValueException.append("DataFrame Xdf has no rows")
        if len(ydf.columns) != 1:
            raise InvalidParameterValueException.append("DataFrame ydf has 0 or more then 1 columns")
        if len(ydf.index) == 0:
            raise InvalidParameterValueException.append("DataFrame ydf has no rows")

        if sklearnMl._estimator_type == "classifier":
            if np.issubdtype(ydf.dtypes[0], np.floating):
                raise InvalidParameterValueException("A classifier model cannot us a float y (target)")
        else:
            if not np.issubdtype(ydf.dtypes[0], np.number):
                raise InvalidParameterValueException("A regressor model can only use number y (target)")

        if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
            raise InvalidParameterValueException("DataFrame ydf or Xdf contains Nodata-values")

        t = permutation_importance(
            sklearnMl,
            Xdf,
            ydf,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
            max_samples=max_samples,
        )
        if importance is None:
            importance = pd.DataFrame(
                zip(fields, t.importances_mean, t.importances_std),
                columns=["feature", "permutation mean", "permutation std"],
            )
        else:
            importance = pd.DataFrame(
                zip(fields, trf, t.importances_mean, t.importances_std),
                columns=["feature", "RandomForest", "permutation mean", "permutation std"],
            )
    return importance


# *******************************
@beartype
def sklearn_model_importance(
    sklearnMl: Any,
    Xdf: Optional[pd.DataFrame] = None,
    ydf: Optional[pd.DataFrame] = None,
    n_repeats: Optional[int] = 5,  # number of permutation, default = 5
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    max_samples: Optional[Union[float, int]] = 1.0,
) -> pd.DataFrame:
    """
       Calculate feature importance.

       without Xdf and ydf:   Importance for RanomForrestClassifier and Regressor
       with Xdf and ydf:      Permutation importance - verry time consuming

    Args:
       - sklearnMl (model): Even for comparison with a testset the model is used to get the model-typ
         (regression or classification)
       - Xdf ("array-like"): subset of X of training
       - ydf ("array-like"): subset of y of training
       for Permutation importances:
       - n_repeats (int, default=5): Number of times to permute a feature:  higher number mean more time!
       - random_state (int, default=None): RandomState instance
          Pseudo-random number generator to control the permutations of each feature.
          Pass an int to get reproducible results across function calls.
       - max_samples (int or float, default=1.0): The number of samples to draw from X to compute feature
         importance in each repeat (without replacement).
          - If int, then draw max_samples samples.
          - If float, then draw max_samples * X.shape[0] samples.
          - If max_samples is equal to 1.0 or X.shape[0], all samples will be used.
          While using this option may provide less accurate importance estimates, it keeps the method tractable
          when evaluating feature importance on large datasets.
          In combination with n_repeats, this allows to control the computational speed vs statistical accuracy
          trade-off of this method.

    Returns:
          Importance of Random Forrest: 1 column
          Permutation importance: 2 colums (mean and std)
    """

    # Argument evaluation
    t = (sklearnMl.__class__.__name__)
    if t not in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "Argument sklearnMl is not one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    if Xdf is not None:
        if ydf is None:
            raise InvalidParameterValueException("Xdf is not Non but ydf is None")
        else:
            if Xdf.shape[0] != ydf.shape[0]:
                raise InvalidParameterValueException("Xdf and ydf have not the same number of rows")
            if len(Xdf.columns) == 0:
                raise InvalidParameterValueException("DataFrame Xdf has no column")
            if len(Xdf.index) == 0:
                raise InvalidParameterValueException("DataFrame Xdf has no rows")
            if len(ydf.columns) != 1:
                raise InvalidParameterValueException("DataFrame ydf has 0 or more then 1 columns")
            if len(ydf.index) == 0:
                raise InvalidParameterValueException("DataFrame ydf has no rows")
    if not hasattr(sklearnMl, "feature_names_in_"):
        raise InvalidParameterValueException("Model is not fitted")

    return _sklearn_model_importance(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        ydf=ydf,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )
