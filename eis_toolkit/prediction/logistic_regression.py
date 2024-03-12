import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.linear_model import LogisticRegression

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.machine_learning_general import _train_and_validate_sklearn_model


@beartype
def logistic_regression_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["accuracy", "precision", "recall", "f1", "auc"]] = ["accuracy"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    penalty: Literal["l1", "l2", "elasicnet", None] = "l2",
    max_iter: int = 100,
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
    verbose: int = 0,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[LogisticRegression, dict]:
    """
    Train and optionally validate a Logistic Regression classifier model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    The choice of the algorithm depends on the penalty chosen. Supported penalties by solver:
    'lbfgs' - ['l2', None]
    'liblinear' - ['l1', 'l2']
    'newton-cg' - ['l2', None]
    'newton-cholesky' - ['l2', None]
    'sag' - ['l2', None]
    'saga' - ['elasticnet', 'l1', 'l2', None]

    For more information about Sklearn Logistic Regression, read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

    Args:
        X: Training data.
        y: Target labels.
        validation_method: Validation method to use. "split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "accuracy".
        split_size: Fraction of the dataset to be used as validation data (rest is used for training).
            Used only when validation_method is "split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when validation_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        penalty: Specifies the norm of the penalty. Defaults to 'l2'.
        max_iter: Maximum number of iterations taken for the solvers to converge. Defaults to 100.
        solver: Algorithm to use in the optimization problem. Defaults to 'lbfgs'.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            values 1 or above will produce prints.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's LogisticRegression.

    Returns:
        The trained Logistric Regression classifier and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if max_iter < 1:
        raise InvalidParameterValueException("Max iter must be > 0.")
    if verbose < 0:
        raise InvalidParameterValueException("Verbose must be a non-negative number.")

    model = LogisticRegression(
        penalty=penalty, max_iter=max_iter, random_state=random_state, solver=solver, verbose=verbose, **kwargs
    )

    model, metrics = _train_and_validate_sklearn_model(
        X=X,
        y=y,
        model=model,
        validation_method=validation_method,
        metrics=metrics,
        split_size=split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics
