"""
creat multilayerperceptron_classifier model
Created an Dezember 01 2022
@author: torchala 
""" 

"""
creat multilayerperceptron_regressor model
Created an November 29 2022
@author: torchala 
""" 
#### Stand:  fast fertig 
#        Tests vor allem mit ESRI-Grids, ob die nodata-Werte korret erkannt wrden
#        Multitarget einführen (ydf.shape.__len__()) und testen. 
#        Weitere Tests für andere Anwendunegn (nicht Raster, sondern shape, csv, ESRI-feature-classes..)
#        Tests mit sample_weights als 3. Argument in fit()
#             # ,sample_weight   # sample_weight: array-like of shape (n_samples,), default=None
#                                    # <= 0:   Sample wird ignoriert
#        Tests. mit weiteren Argumenten des Sklaern-Moduls

from typing import Optional, Literal
from sklearn.neural_network import MLPClassifier
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
activation = Literal["identity", "logistic", "tanh", "relu"]
solver = Literal["lbfgs", "sgd", "adam"]
learning = Literal["constant", "invscaling", "adaptive"]
def _mlp_classifier(  # type: ignore[no-any-unimported]
    hidden_layer_sizes: Optional[tuple] =(100,),
    activation: Optional[activation] = "relu",
    solver: Optional[activation]="adam",
    alpha: Optional[float] = 0.0001,
    batch_size: Optional [int | str] ="auto",
    learning_rate: Optional[learning] = "constant",
    learning_rate_init: Optional[float] = 0.001,
    power_t: Optional[float] = 0.5,
    max_iter: Optional[int] = 200,
    shuffle: Optional[bool] = True,
    random_state: Optional[int] = None,  # alternatively: RandomState instance
    tol: Optional[float] = 1e-4,
    verbose: Optional[bool] =False,
    warm_start: Optional[bool]=False,
    momentum: Optional[float]=0.9,
    nesterovs_momentum: Optional[bool]=True,
    early_stopping: Optional[bool]=False,
    validation_fraction: Optional[float] =0.1,
    beta_1: Optional[float] =0.9,
    beta_2: Optional[float] =0.999,
    epsilon: Optional[float] =1e-8,
    n_iter_no_change: Optional[int]=10,
    max_fun: Optional[int]=15000
):

    myML = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,
        activation = activation,
        solver = solver,
        alpha = alpha,
        batch_size = batch_size, 
        learning_rate = learning_rate,
        learning_rate_init = learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun
        )

    return myML

# *******************************
activation = Literal["identity", "logistic", "tanh", "relu"]
solver = Literal["lbfgs", "sgd", "adam"]
learning = Literal["constant", "invscaling", "adaptive"]
def mlp_classifier(  # type: ignore[no-any-unimported]
    hidden_layer_sizes: Optional[tuple] =(100,),
    activation: Optional[activation] = "relu",
    solver: Optional[activation]="adam",
    alpha: Optional[float] = 0.0001,
    batch_size: Optional [int | str] ="auto",
    learning_rate: Optional[learning] = "constant",
    learning_rate_init: Optional[float] = 0.001,
    power_t: Optional[float] = 0.5,
    max_iter: Optional[int] = 200,
    shuffle: Optional[bool] = True,
    random_state: Optional[int] = None,  # alternatively: RandomState instance
    tol: Optional[float] = 1e-4,
    verbose: Optional[bool] =False,
    warm_start: Optional[bool]=False,
    momentum: Optional[float]=0.9,
    nesterovs_momentum: Optional[bool]=True,
    early_stopping: Optional[bool]=False,
    validation_fraction: Optional[float] =0.1,
    beta_1: Optional[float] =0.9,
    beta_2: Optional[float] =0.999,
    epsilon: Optional[float] =1e-8,
    n_iter_no_change: Optional[int]=10,
    max_fun: Optional[int]=15000
):

    """ training of a multilayer percetron (MLP) classification model

    Args:
    hidden_layer_sizes tuple, length = n_layers 2, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation {'identity", "logistic", "tanh", "relu"}, default="relu"
       Activation function for the hidden layer.
        "identity", no-op activation, useful to implement linear bottleneck, returns f(x) = x
        "logistic", the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        "tanh", the hyperbolic tan function, returns f(x) = tanh(x).
        "relu", the rectified linear unit function, returns f(x) = max(0, x)
    -solver {"lbfgs", "sgd", "adam"}, default="adam"
        The solver for weight optimization.
        "lbfgs" is an optimizer in the family of quasi-Newton methods.
        "sgd" refers to stochastic gradient descent.
        "adam" refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
        Note: The default solver "adam" works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, "lbfgs" can converge faster and perform better.
    alpha float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss.
    batch_size int, default="auto"
        Size of minibatches for stochastic optimizers. If the solver is "lbfgs", the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples).
    learning_rate {"constant", "invscaling", "adaptive"}, default="constant"
        Learning rate schedule for weight updates.
        "constant" is a constant learning rate given by "learning_rate_init".
        "invscaling" gradually decreases the learning rate learning_rate_ at each time step "t" using an inverse scaling exponent of "power_t". effective_learning_rate = learning_rate_init / pow(t, power_t)
        "adaptive" keeps the learning rate constant to "learning_rate_init" as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if "early_stopping" is on, the current learning rate is divided by 5.
        Only used when solver="sgd".
    learning_rate_init float, default=0.001
        The initial learning rate used. It controls the step-size in updating the weights. Only used when solver="sgd" or "adam".
    power_t float, default=0.5
        The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to "invscaling". Only used when solver="sgd".
    max_iter int, default=200
        Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers ("sgd", "adam"), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
    shuffle bool, default=True
        Whether to shuffle samples in each iteration. Only used when solver="sgd" or "adam".
    random_state int, RandomState instance, default=None
        Determines random number generation for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver="sgd" or "adam". Pass an int for reproducible results across multiple function calls. See Glossary.
    tol float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to "adaptive", convergence is considered to be reached and training stops.
    verbose bool, default=False
        Whether to print progress messages to stdout.
    warm_start bool, default=False
        When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. See the Glossary.
    momentum float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only used when solver="sgd".
    nesterovs_momentum bool, default=True
        Whether to use Nesterov's momentum. Only used when solver="sgd" and "momentum" > 0.
    early_stopping bool, default=False
        Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver="sgd" or "adam".
     validation_fraction float, default=0.1
        The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.
    beta_1 float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver="adam".
    beta_2 float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver="adam".
    epsilon float, default=1e-8
        Value for numerical stability in adam. Only used when solver="adam".
    n_iter_no_change int, default=10
        Maximum number of epochs to not meet tol improvement. Only effective when solver="sgd" or "adam".
    max_fun int, default=15000
        Only used when solver="lbfgs". Maximum number of function calls. The solver iterates until convergence (determined by "tol"), number of iterations reaches max_iter, or this number of function calls. Note that number of function calls will be greater than or equal to the number of iterations for the MLPRegressor.

    Returns:
        mlp classification model
    """

    myML = _mlp_classifier( 
        hidden_layer_sizes = hidden_layer_sizes,
        activation = activation,
        solver = solver,
        alpha = alpha,
        batch_size = batch_size, 
        learning_rate = learning_rate,
        learning_rate_init = learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun
    )

    return myML

