from typing import Literal

import sklearn
from beartype import beartype
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

from eis_toolkit.exceptions import InvalidCrossValidationSelected, InvalidNumberOfSplit


@beartype
def performance_model_estimation(
    cross_validation_type: Literal["LOOCV", "KFOLD", "SKFOLD"], number_of_split: int = 5
) -> sklearn.model_selection:
    """
    Evaluate the feature importance of a sklearn classifier or linear model.

    Parameters:
        cross_validation_type: Select cross validation (LOOCV, SKFOLD, KFOLD).
        number_of_split: number used to split the dataset.
    Return:
        Selected cross validation method
    Raises:
        InvalidCrossValidationSelected: When the cross validation method selected is not implemented.
        InvalidNumberOfSplit: When the number of split is incompatible with the selected cross validation
    """

    if cross_validation_type is None:
        raise InvalidCrossValidationSelected

    if cross_validation_type != "LOOCV" and number_of_split <= 1:
        raise InvalidNumberOfSplit
    if cross_validation_type == "LOOCV":
        return LeaveOneOut()
    elif cross_validation_type == "KFOLD":
        return KFold(n_splits=number_of_split, shuffle=True)
    elif cross_validation_type == "SKFOLD":
        return StratifiedKFold(n_splits=number_of_split, shuffle=True)
    else:
        raise InvalidCrossValidationSelected
