from beartype import beartype
from beartype.typing import Literal, Optional, Union, Any

from sklearn.linear_model import LogisticRegression

# *******************************
penalty = Literal["l1", "l2", "elasticnet", "none"]
solver = Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
balanced = Literal["balanced"]


@beartype
def _sklearn_logistic_regression(
    penalty: Optional[penalty] = "l2",
    dual: Optional[bool] = False,
    tol: Optional[Union[float, int]] = 1e-4,
    C: Optional[Union[float, int]] = 1,
    fit_intercept: Optional[bool] = True,
    intercept_scaling: Optional[Union[int, float]] = 1,
    class_weight: Optional[Union[dict, balanced]] = None,
    random_state: Optional[int] = None,
    solver: Optional[solver] = "lbfgs",
    max_iter: Optional[int] = 100,
    verbose: Optional[int] = 0,
    warm_start: Optional[bool] = False,
    n_jobs: Optional[int] = None,
    l1_ratio: Optional[Union[float, int]] = None,
) -> Any:

    sklearnMl = LogisticRegression(
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )

    return sklearnMl


# *******************************
penalty = Literal["l1", "l2", "elasticnet", "none"]
solver = Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
balanced = Literal["balanced"]


@beartype
def sklearn_logistic_regression(
    penalty: Optional[penalty] = "l2",
    dual: Optional[bool] = False,
    tol: Optional[Union[float, int]] = 1e-4,
    C: Optional[Union[float, int]] = 1,
    fit_intercept: Optional[bool] = True,
    intercept_scaling: Optional[Union[float, int]] = 1,
    class_weight: Optional[Union[dict, balanced]] = None,
    random_state: Optional[int] = None,
    solver: Optional[solver] = "lbfgs",
    max_iter: Optional[int] = 100,
    verbose: Optional[int] = 0,
    warm_start: Optional[bool] = False,
    n_jobs: Optional[int] = None,
    l1_ratio: Optional[Union[float, int]] = None,
) -> Any:

    """
        creating of a logistic regression model
    Args:
        - penalty ({'l1', 'l2', 'elasticnet', 'none'}, default='l2'):
            Specify the norm of the penalty:
            'none': no penalty is added;
            'l2': add a L2 penalty term and it is the default choice;
            'l1': add a L1 penalty term;
            'elasticnet': both L1 and L2 penalty terms are added.
            Warning Some penalties may not work with some solvers.
            See the parameter solver below, to know the compatibility between the penalty and solver.
        - dual (default=False):
            Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.
        - tol (default=1e-4):
            Tolerance for stopping criteria.
        - C (default=1.0):
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger regularization.
        - fit_intercept (default=True):
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        - intercept_scaling (float, default=1):
            Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector.
            The intercept becomes intercept_scaling * synthetic_feature_weight.
            Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
        - class_weight (dict or 'balanced', default=None):
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        - random_state (RandomState instance, default=None):
            Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. See Glossary for details.
        - solver ({'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'):
            Algorithm to use in the optimization problem. Default is 'lbfgs'. To choose a solver, you might want to consider the following aspects:
            For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones;
            For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss;
            'liblinear' is limited to one-versus-rest schemes.
            Warning The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:
            'newton-cg' - ['l2', 'none']
            'lbfgs' - ['l2', 'none']
            'liblinear' - ['l1', 'l2']
            'sag' - ['l2', 'none']
            'saga' - ['elasticnet', 'l1', 'l2', 'none']
            Note 'sag' and 'saga' fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
            See also Refer to the User Guide for more information regarding LogisticRegression and more specifically the Table summarizing solver/penalty supports.
        - max_iter (default=100):
            Maximum number of iterations taken for the solvers to converge.
       - class_weight ({“balanced”, “balanced_subsample”}, default=None):
          Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
          The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
          The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
          Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        - verbose (default=0):
            For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
        - warm_start (default=False):
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary.
        - n_jobs (default=None):
            Number of CPU cores used when parallelizing over classes if multi_class='ovr'”. This parameter is ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
        - l1_ratio (default=None):
            The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
    Returns:
        SKLEARN model (logistic regression)
    """

    sklearnMl = _sklearn_logistic_regression(
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )

    return sklearnMl
