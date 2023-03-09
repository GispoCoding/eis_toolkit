
from typing import Optional, Literal
from sklearn.linear_model import LogisticRegression
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
penalty = Literal['l1','l2','elasticnet','none']
solver = Literal['newton-cg','lbfgs','liblinear','sag','saga']
balanced = Literal['balanced']
def _sklearn_logistic_regression(
    penalty: Optional [penalty] = 'l2',
    dual: Optional [bool] = False,
    tol: Optional [float | int] = 1e-4,
    C: Optional [float | int] = 1,
    fit_intercept: Optional [bool] = True,
    intercept_scaling: Optional [int | float] = 1,
    class_weight: Optional [dict | balanced] = None,
    random_state: Optional [int] = None,
    solver: Optional[solver] = 'lbfgs',
    max_iter: Optional [int] = 100,
    verbose: Optional [int] = 0,
    warm_start: Optional [bool] = False,
    n_jobs: Optional [int] = None,
    l1_ratio: Optional [float | int] = None,
):

    sklearnMl = LogisticRegression( 
        penalty = penalty,
        dual = dual,
        tol = tol,
        C = C,
        fit_intercept = fit_intercept,
        intercept_scaling = intercept_scaling,
        class_weight = class_weight,
        random_state = random_state,
        solver = solver,
        max_iter = max_iter,
        verbose = verbose,
        warm_start = warm_start,
        n_jobs = n_jobs,
        l1_ratio = l1_ratio,
    )

    return sklearnMl

# *******************************
penalty = Literal['l1','l2','elasticnet','none']
solver = Literal ['newton-cg','lbfgs','liblinear','sag','saga']
balanced = Literal ['balanced']
def sklearn_logistic_regression(  # type: ignore[no-any-unimported]
    penalty: Optional [penalty] = 'l2',
    dual: Optional [bool] = False,
    tol: Optional [float] = 1e-4,
    C: Optional [float] = 1,
    fit_intercept: Optional [bool] = True,
    intercept_scaling: Optional [float] = 1,
    class_weight: Optional [dict | balanced] = None,
    random_state: Optional [int] = None,
    solver: Optional[solver] = 'lbfgs',
    max_iter: Optional [int] = 100,
    verbose: Optional [int] = 0,
    warm_start: Optional [bool] = False,
    n_jobs: Optional [int] = None,
    l1_ratio: Optional [float] = None,
):

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
        - dual (bool, default=False):
            Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. 
            Prefer dual=False when n_samples > n_features.
        - tol (float, default=1e-4):
            Tolerance for stopping criteria.
        - C (float, default=1.0):
            Inverse of regularization strength; must be a positive float. 
            Like in support vector machines, smaller values specify stronger regularization.
        - fit_intercept (bool, default=True):
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        - intercept_scaling (float, default=1):
            Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. 
            The intercept becomes intercept_scaling * synthetic_feature_weight.
            Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
        - class_weight (dict or 'balanced', default=None):
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        - random_state (int, RandomState instance, default=None):
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
        - max_iter (int, default=100):
            Maximum number of iterations taken for the solvers to converge.
        - multi_class ({'auto', 'ovr', 'multinomial'}, default='auto'):
            If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'.
        - verbose (int, default=0):
            For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
        - warm_start (bool, default=False):
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary.
        - n_jobs (int, default=None):
            Number of CPU cores used when parallelizing over classes if multi_class='ovr'”. This parameter is ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
        - l1_ratio (float, default=None):
            The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
    Returns:
        SKLEARN model (logistic regression)
    """

    # Argument evaluation
    fl = []
    if penalty is not None:
        if not (penalty in ['l1','l2','elasticnet','none']):
            fl.append('Argument penalty is not in (l1,l2,elasticnet,none)')
    if solver is not None:
        if not (solver in ['newton-cg','lbfgs','liblinear','sag','saga']):
            fl.append('Argument solver is not in (newton-cg,lbfgs,liblinear,sag,saga)')
    if class_weight is not None:
        if not ((class_weight in ['balanced']) or isinstance(class_weight,dict)):
            fl.append('Argument balanced is not in (balanced) not dictionary and not None')
    if not (isinstance(random_state, int) or (random_state is None)):
        fl.append('Argument random_state is not integer and is not None')
    if not (isinstance(max_iter, int) or (max_iter is None)):
        fl.append('Argument max_iter is not integer and is not None')
    if not (isinstance(n_jobs,int) or (n_jobs is None)):
        fl.append('Argument n_jobs is not integer and is not None')
    if not (isinstance(verbose, int) or (verbose is None)):
        fl.append('Argument verbose is not integer and is not None')
    if not (isinstance(tol, (int,float)) or (tol is None)):
        fl.append('Argument tol is not float and is not None')
    if not (isinstance(C, (int,float)) or (C is None)):
        fl.append('Argument C is not float and is not None')
    if not (isinstance(intercept_scaling,(int,float)) or (intercept_scaling is None)):
        fl.append('Argument intercept_scaling is not float and is not None')
    if not (isinstance(l1_ratio, (int,float)) or (l1_ratio is None)):
        fl.append('Argument l1_ratio is not float and is not None')
    if not (isinstance(dual, bool) or (dual is None)):
        fl.append('Argument dual is not bool and is not None')
    if not (isinstance(fit_intercept, bool) or (fit_intercept is None)):
        fl.append('Argument fit_intercept is not bool and is not None')
    if not (isinstance(warm_start, bool) or (warm_start is None)):
        fl.append('Argument warm_start is not bool and is not None')    
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])
    
    if solver in ['lbfgs','newton_cg','newton_cholesky','sag']:
        if not (penalty in ['l2','none']):
            raise InvalidParameterValueException ('For solver argument penalty should be l2 or none')
    elif solver == 'liblinear':
        if not (penalty in ['l2','l1']):
            raise InvalidParameterValueException ('For solver argument penalty should be l2 or l1')
            # solver - penalty: 
            # ‘lbfgs’ - [‘l2’, None]
            # ‘liblinear’ - [‘l1’, ‘l2’]
            # ‘newton-cg’ - [‘l2’, None]
            # ‘newton-cholesky’ - [‘l2’, None]
            # ‘sag’ - [‘l2’, None]
            # ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, None]

    sklearnMl = _sklearn_logistic_regression( 
        penalty = penalty,
        dual = dual,
        tol = tol,
        C  = C,
        fit_intercept  = fit_intercept,
        intercept_scaling = intercept_scaling,
        class_weight = class_weight,
        random_state = random_state,
        solver = solver,
        max_iter = max_iter,
        verbose = verbose,
        warm_start = warm_start,
        n_jobs = n_jobs,
        l1_ratio = l1_ratio,
    )

    return sklearnMl
    