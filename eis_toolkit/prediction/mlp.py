from numbers import Number

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple
from tensorflow import keras
from tensorflow.keras.optimizers.legacy import SGD, Adagrad, Adam, RMSprop

from eis_toolkit.exceptions import InvalidDataShapeException, InvalidParameterValueException


def _keras_optimizer(optimizer: str, **kwargs):
    """Create a Keras optimizer from given name and parameters."""
    if optimizer == "adam":
        return Adam(**kwargs)
    elif optimizer == "adagrad":
        return Adagrad(**kwargs)
    elif optimizer == "rmsprop":
        return RMSprop(**kwargs)
    elif optimizer == "sgd":
        return SGD(**kwargs)
    else:
        raise InvalidParameterValueException(f"Unidentified optimizer: {optimizer}")


def _check_MLP_inputs(
    neurons: Sequence[int],
    validation_split: Optional[float],
    learning_rate: float,
    dropout_rate: Optional[Number],
    es_patience: int,
    batch_size: int,
    epochs: int,
    output_neurons: int,
    loss_function: str,
) -> None:
    """Check parameters for Keras MLP training."""
    if len(neurons) == 0:
        raise InvalidParameterValueException("Neurons parameter must be a non-empty list.")

    if any(neuron < 1 for neuron in neurons):
        raise InvalidParameterValueException("Each neuron in neurons list must be at least 1.")

    if validation_split and not (0.0 < validation_split < 1.0):
        raise InvalidParameterValueException("Validation split must be a value between 0 and 1, exclusive.")

    if learning_rate <= 0.0:
        raise InvalidParameterValueException("Learning rate must be greater than 0.")

    if dropout_rate and not (0.0 <= dropout_rate <= 1.0):
        raise InvalidParameterValueException("Dropout rate must be between 0 and 1, inclusive.")

    if es_patience <= 0:
        raise InvalidParameterValueException("Early stopping patience must be greater than 0.")

    if batch_size <= 0:
        raise InvalidParameterValueException("Batch size must be greater than 0.")

    if epochs <= 0:
        raise InvalidParameterValueException("Number of epochs must be greater than 0.")

    if output_neurons <= 0:
        raise InvalidParameterValueException("Number of output neurons must be greater than 0.")

    if output_neurons > 1 and loss_function == "binary_crossentropy":
        raise InvalidParameterValueException(
            "Number of output neurons must be 1 when used loss function is binary crossentropy."
        )

    if output_neurons <= 2 and loss_function == "categorical_crossentropy":
        raise InvalidParameterValueException(
            "Number of output neurons must be greater than 2 when used loss function is categorical crossentropy."
        )


def _check_ML_model_data_input(X: np.ndarray, y: np.ndarray):
    """Check if the input data for the ML model is in the correct shape."""
    if X.ndim != 2:
        raise InvalidDataShapeException(f"X must be a 2-dimensional array, but is an array with shape {X.shape}.")

    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0]

    if n_samples_X != n_samples_y:
        raise InvalidDataShapeException(
            f"The number of samples in X and y must be equal, but got {n_samples_X} in X and {n_samples_y} in y."
        )


@beartype
def train_MLP_classifier(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    activation: Literal["relu", "linear", "sigmoid", "tanh"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam", "adagrad", "rmsprop", "sdg"] = "adam",
    learning_rate: Number = 0.001,
    loss_function: Literal["binary_crossentropy", "categorical_crossentropy"] = "binary_crossentropy",
    dropout_rate: Optional[Number] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["accuracy", "precision", "recall", "f1_score"]]] = ["accuracy"],
    random_state: Optional[int] = None,
) -> Tuple[keras.Model, dict]:
    """
    Train MLP (Multilayer Perceptron) using Keras.

    Creates a Sequential model with Dense NN layers. For each element in `neurons`, Dense layer with corresponding
    dimensionality/neurons is created with the specified activation function (`activation`). If `dropout_rate` is
    specified, a Dropout layer is added after each Dense layer.

    Parameters default to a binary classification model using sigmoid as last activation, binary crossentropy as loss
    function and 1 output neuron/unit.

    For more information about Keras models, read the documentation here: https://keras.io/.

    Args:
        X: Input data. Should be a 2-dimensional array where each row represents a sample and each column a
            feature. Features should ideally be normalized or standardized.
        y: Target labels. For binary classification, y should be a 1-dimensional array of binary labels (0 or 1).
            For multi-class classification, y should be a 2D array with one-hot encoded labels. The number of columns
            should match the number of classes.
        neurons: Number of neurons in each hidden layer.
        validation_split: Fraction of data used for validation during training. Value must be > 0 and < 1 or None.
            Defaults to 0.2.
        validation_data: Separate dataset used for validation during training. Overrides validation_split if
            provided. Expected data form is (X_valid, y_valid). Defaults to None.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer. Defaults to 1.
        last_activation: Activation function used in the output layer. Defaults to 'sigmoid'.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Value must be > 0. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'binary_crossentropy'.
        dropout_rate: Fraction of the input units to drop. Value must be >= 0 and <= 1. Defaults to None.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['accuracy'].
        random_state: Seed for random number generation. Sets Python, Numpy and Tensorflow seeds to make
            program deterministic. Defaults to None (random state / seed).

    Returns:
        Trained MLP model and training history.

    Raises:
        InvalidParameterValueException: Some of the numeric parameters have invalid values.
        InvalidDataShapeException: Shape of X or y is invalid.
    """
    # 1. Check input data
    _check_ML_model_data_input(X=X, y=y)
    _check_MLP_inputs(
        neurons=neurons,
        validation_split=validation_split,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        es_patience=es_patience,
        batch_size=batch_size,
        epochs=epochs,
        output_neurons=output_neurons,
        loss_function=loss_function,
    )

    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    # 2. Create and compile a sequential model
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(X.shape[1],)))

    for neuron in neurons:
        model.add(keras.layers.Dense(units=neuron, activation=activation))

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(units=output_neurons, activation=last_activation))

    model.compile(
        optimizer=_keras_optimizer(optimizer, learning_rate=learning_rate), loss=loss_function, metrics=metrics
    )

    # 3. Train the model
    # Early stopping callback
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []

    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=validation_split if validation_split else 0.0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history.history


@beartype
def train_MLP_regressor(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    activation: Literal["relu", "linear", "sigmoid", "tanh"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["linear"] = "linear",
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam", "adagrad", "rmsprop", "sdg"] = "adam",
    learning_rate: Number = 0.001,
    loss_function: Literal["mse", "mae", "hinge", "huber"] = "mse",
    dropout_rate: Optional[Number] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["mse", "rmse", "mae"]]] = ["mse"],
    random_state: Optional[int] = None,
) -> Tuple[keras.Model, dict]:
    """
    Train MLP (Multilayer Perceptron) using Keras.

    Creates a Sequential model with Dense NN layers. For each element in `neurons`, Dense layer with corresponding
    dimensionality/neurons is created with the specified activation function (`activation`). If `dropout_rate` is
    specified, a Dropout layer is added after each Dense layer.

    For more information about Keras models, read the documentation here: https://keras.io/.

    Args:
        X: Input data. Should be a 2-dimensional array where each row represents a sample and each column a
            feature. Features should ideally be normalized or standardized.
        y: Target labels. Should be a 1-dimensional array where each entry corresponds to the continuous
            target value for the respective sample in X.
        neurons: Number of neurons in each hidden layer.
        validation_split: Fraction of data used for validation during training. Value must be > 0 and < 1 or None.
            Defaults to 0.2.
        validation_data: Separate dataset used for validation during training. Overrides validation_split if
            provided. Expected data form is (X_valid, y_valid). Defaults to None.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer. Defaults to 1.
        last_activation: Activation function used in the output layer. Defaults to 'linear'.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Value must be > 0. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'mse'.
        dropout_rate: Fraction of the input units to drop. Value must be >= 0 and <= 1. Defaults to None.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['mse'].
        random_state: Seed for random number generation. Sets Python, Numpy and Tensorflow seeds to make
            program deterministic. Defaults to None (random state / seed).

    Returns:
        Trained MLP model and training history.

    Raises:
        InvalidParameterValueException: Some of the numeric parameters have invalid values.
        InvalidDataShapeException: Shape of X or y is invalid.
    """
    # 1. Check input data
    _check_ML_model_data_input(X=X, y=y)
    _check_MLP_inputs(
        neurons=neurons,
        validation_split=validation_split,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        es_patience=es_patience,
        batch_size=batch_size,
        epochs=epochs,
        output_neurons=output_neurons,
        loss_function=loss_function,
    )

    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    # 2. Create and compile a sequential model
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(X.shape[1],)))

    for neuron in neurons:
        model.add(keras.layers.Dense(units=neuron, activation=activation))

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(units=output_neurons, activation=last_activation))

    model.compile(
        optimizer=_keras_optimizer(optimizer, learning_rate=learning_rate), loss=loss_function, metrics=metrics
    )

    # 3. Train the model
    # Early stopping callback
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []

    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=validation_split if validation_split else 0.0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history.history
