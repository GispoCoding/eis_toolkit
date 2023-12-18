from numbers import Number
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from beartype import beartype
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from tensorflow import keras

from eis_toolkit import exceptions


def _keras_optimizer(optimizer: str, **kwargs):
    if optimizer == "adam":
        return Adam(**kwargs)
    elif optimizer == "adagrad":
        return Adagrad(**kwargs)
    elif optimizer == "rmsprop":
        return RMSprop(**kwargs)
    elif optimizer == "sdg":
        return SGD(**kwargs)
    else:
        raise exceptions.InvalidParameterValueException(f"Unidentified optimizer: {optimizer}")


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
    """Check inputs for MLP."""
    if len(neurons) == 0:
        raise exceptions.InvalidParameterValueException("Neurons parameter must be a non-empty list.")

    if any(neuron < 1 for neuron in neurons):
        raise exceptions.InvalidParameterValueException("Each neuron in neurons list must be at least 1.")

    if validation_split and not (0 < validation_split < 1):
        raise exceptions.InvalidParameterValueException("Validation split must be a value between 0 and 1, exclusive.")

    if learning_rate <= 0:
        raise exceptions.InvalidParameterValueException("Learning rate must be greater than 0.")

    if dropout_rate and not (0 <= dropout_rate <= 1):
        raise exceptions.InvalidParameterValueException("Dropout rate must be between 0 and 1, inclusive.")

    if es_patience <= 0:
        raise exceptions.InvalidParameterValueException("Early stopping patience must be greater than 0.")

    if batch_size <= 0:
        raise exceptions.InvalidParameterValueException("Batch size must be greater than 0.")

    if epochs <= 0:
        raise exceptions.InvalidParameterValueException("Number of epochs must be greater than 0.")

    if output_neurons <= 0:
        raise exceptions.InvalidParameterValueException("Number of output neurons must be greater than 0.")

    if output_neurons > 1 and loss_function == "binary_crossentropy":
        raise exceptions.InvalidParameterValueException(
            "Number of output neurons must be 1 when used loss function is binary crossentropy."
        )


@beartype
def train_MLP_classifier(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int] = [16],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    activation: Literal["relu", "linear", "sigmoid", "tanh"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam", "adagrad", "rmsprop", "sdg"] = "adam",
    learning_rate: Number = 0.001,
    loss_function: Literal["categorical_crossentropy", "binary_crossentropy"] = "categorical_crossentropy",
    dropout_rate: Optional[Number] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["accuracy", "precision", "recall"]]] = ["accuracy"],
    random_state: Optional[int] = None,
) -> Tuple[keras.Model, dict]:
    """
    Train MLP (Multilayer Perceptron) using Keras.

    Creates a Sequential model with Dense NN layers. For each element in `neurons`, Dense layer with corresponding
    dimensionality/neurons is created with the specified activation function (`activation`). If `dropout_rate` is
    specified, a Dropout layer is added after each Dense layer. In the end, a Flatten layer is added followed by
    the output layer (Dense that uses `output_neruons` and `last_activation`).

    Args:
        X: Input data.
        y: Target labels.
        neurons: Number of neurons in each hidden layer. Defaults to one layer with 16 neurons.
        validation_split: Fraction of data used for validation during training. Value must be > 0 and < 1 or None.
            Defaults to 0.2.
        validation_data: Separate dataset used for validation during training. Overrides validation_split if
            provided. Expected data form is (X_valid, y_valid). Defaults to None.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer. Defaults to 1.
        last_activation: Activation function used in the output layer. Defaults to 'softmax'.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Value must be > 0. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'categorical_crossentropy'.
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
    """
    # 1. Check input data
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
        validation_split=validation_split if validation_split else 0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history.history


@beartype
def train_MLP_regressor(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int] = [16],
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
    specified, a Dropout layer is added after each Dense layer. In the end, a Flatten layer is added followed by
    the output layer (Dense that uses `output_neruons` and `last_activation`).

    Args:
        X: Input data.
        y: Target labels.
        neurons: Number of neurons in each hidden layer. Defaults to one layer with 16 neurons.
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
    """
    # 1. Check input data
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
        validation_split=validation_split if validation_split else 0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history.history


# NOTE: Old, currently unused code. Can be deleted if not needed.

# def check_keras_training_arguments(func):
#     """Check inputs to _train_and_validate_MLP and _train_and_validate_CNN."""

#     @wraps(func)
#     def decorated_func(*args, **kwargs):

#         neurons = kwargs.get("neurons")
#         if len(neurons) == 0:
#             raise exceptions.InvalidParameterValueException("Neurons parameter must be a non-empty list.")

#         test_split = kwargs.get("test_split")
#         if not (0 < test_split < 1):
#             raise exceptions.InvalidParameterValueException("Test split must be a value between 0 and 1, exclusive.")

#         learning_rate = kwargs.get("learning_rate")
#         if learning_rate <= 0:
#             raise exceptions.InvalidParameterValueException("Learning rate must be greater than 0.")

#         dropout_rate = kwargs.get("dropout_rate")
#         if not (0 <= dropout_rate <= 1):
#             raise exceptions.InvalidParameterValueException("Dropout rate must be between 0 and 1, inclusive.")

#         es_patience = kwargs.get("es_patience")
#         if es_patience <= 0:
#             raise exceptions.InvalidParameterValueException("Early stopping patience must be greater than 0.")

#         batch_size = kwargs.get("batch_size")
#         if batch_size <= 0:
#             raise exceptions.InvalidParameterValueException("Batch size must be greater than 0.")

#         epochs = kwargs.get("epochs")
#         if epochs <= 0:
#             raise exceptions.InvalidParameterValueException("Number of epochs must be greater than 0.")

#         output_neurons = kwargs.get("output_neurons")
#         if output_neurons <= 0:
#             raise exceptions.InvalidParameterValueException("Number of output neurons must be greater than 0.")

#         loss = kwargs.get("loss_function")
#         if output_neurons > 1 and loss == "binary_crossentropy":
#             raise exceptions.InvalidParameterValueException(
#                 "Number of output neurons must be 1 when used loss function is binary crossentropy.")

#         # Continue with the function
#         result = func(*args, **kwargs)
#         return result

#     return decorated_func


# @beartype
# def _train_and_validate(
#     X: np.ndarray,
#     y: np.ndarray,
#     model: keras.Sequential,
#     validation_split: float,
#     test_split: float,
#     output_neurons: int,
#     last_activation: str,
#     kernel_regularizer: keras.regularizers,
#     epochs: int,
#     batch_size: int,
#     optimizer: str,
#     learning_rate: float,
#     loss_function: str,
#     early_stopping: bool,
#     es_patience: int,
#     metrics: Optional[Sequence[str]],
#     random_state: Optional[int] = None,
# ) -> Tuple[keras.Sequential, dict, list]:

#     model.add(keras.layers.Flatten())

#     model.add(
#         keras.layers.Dense(
#             units=output_neurons,
#             activation=last_activation,
#             kernel_regularizer=kernel_regularizer,
#             bias_regularizer=None,
#         )
#     )

#     # Compile the model
# model.compile(
#     optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
#     loss=loss_function,
#     metrics=metrics
# )

#     # Early stopping callback
#     callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []

#     # Separate test data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)

#     # Validation split should be the defined fraction of the whole dataset before test split
#     validation_split = validation_split / (1 - test_split)

#     # Train the model
#     history = model.fit(
#         X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, callbacks=callbacks
#     )

#     # Evaluate the model using test data
#     evaluation = model.evaluate(X_test, y_test)

#     return model, history.history, evaluation
