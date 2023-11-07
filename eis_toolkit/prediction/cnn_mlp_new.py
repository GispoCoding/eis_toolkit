from functools import wraps
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from beartype import beartype
from sklearn.model_selection import train_test_split
from tensorflow import keras

from eis_toolkit import exceptions

# TODO
# 1. Multimodal not implemented yet
# 2. Probabilistic MLP ?
# 3. Hyperparameter optimization:
# - Keras tuner
# - Wrapping model as KerasClassifier and use Sklearn searches
# - Other libraries?
# 4. Cross-validation:
# -
# 5. Visualization
# - Tensorboard?
# - Plot set of graphs at the end

# NOTES:
# 1. Which optimizers are relevant?
# 2. Defaults values ok?
# 3. Sensible set of parameters exposed for the user? Do we want to try add **kwargs for extra inputs?
# 4. Train-validation-test data splitting ok, do we need option to give separate datasets as input?
# 5.


OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
}


# --- Inner functions, utils etc. ---


def check_keras_training_arguments(func):
    """Check inputs to _train_and_validate_MLP and _train_and_validate_CNN."""

    @wraps(func)
    def decorated_func(*args, **kwargs):
        # Check certain inputs
        neurons = kwargs.get("neurons")
        if len(neurons) == 0:
            raise exceptions.InvalidParameterValueException("Neurons parameter must be a non-empty list.")

        test_split = kwargs.get("test_split")
        if not (0 < test_split < 1):
            raise exceptions.InvalidParameterValueException("Test split must be a value between 0 and 1, exclusive.")

        learning_rate = kwargs.get("learning_rate")
        if learning_rate <= 0:
            raise exceptions.InvalidParameterValueException("Learning rate must be greater than 0.")

        dropout_rate = kwargs.get("dropout_rate")
        if not (0 <= dropout_rate <= 1):
            raise exceptions.InvalidParameterValueException("Dropout rate must be between 0 and 1, inclusive.")

        es_patience = kwargs.get("es_patience")
        if es_patience <= 0:
            raise exceptions.InvalidParameterValueException("Early stopping patience must be greater than 0.")

        batch_size = kwargs.get("batch_size")
        if batch_size <= 0:
            raise exceptions.InvalidParameterValueException("Batch size must be greater than 0.")

        epochs = kwargs.get("epochs")
        if epochs <= 0:
            raise exceptions.InvalidParameterValueException("Number of epochs must be greater than 0.")

        # Continue with the function
        result = func(*args, **kwargs)
        return result

    return decorated_func


@beartype
def _train_and_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: keras.Sequential,
    validation_split: float,
    test_split: float,
    output_neurons: int,
    last_activation: str,
    kernel_regularizer: keras.regularizers,
    epochs: int,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    loss_function: str,
    early_stopping: bool,
    es_patience: int,
    metrics: Optional[Sequence[str]],
    random_state: Optional[int] = None,
) -> Tuple[keras.Sequential, dict, list]:

    model.add(keras.layers.Flatten())

    model.add(
        keras.layers.Dense(
            units=output_neurons,
            activation=last_activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=None,
        )
    )

    # Compile the model
    model.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate), loss=loss_function, metrics=metrics)

    # Early stopping callback
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)

    # Validation split should be the defined fraction of the whole dataset before test split
    validation_split = validation_split / (1 - test_split)

    # Train the model
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, callbacks=callbacks
    )

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)

    return model, history.history, evaluation


# --- Public functions ---


@check_keras_training_arguments
@beartype
def train_and_validate_MLP(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int] = [16],
    validation_split: float = 0.15,
    test_split: float = 0.15,
    activation: Literal["relu"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    kernel_regularizer: Optional[Literal["l1", "l2", "l1_l2"]] = None,
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam"] = "adam",
    learning_rate: float = 0.001,
    loss_function: Literal["categorical_crossentropy", "binary_crossentropy", "mse"] = "categorical_crossentropy",
    dropout_rate: Optional[float] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["accuracy", "precision", "recall"]]] = [
        "accuracy"
    ],  # NOTE: Is this a useful parameter? Should there be more options?
) -> Tuple[keras.Sequential, dict, list]:
    """
    Train and validate MLP (Multilayer Perceptron) using Keras.

    Args:
        X: Input data.
        y: Target labels.
        neurons: Number of neurons in each hidden layer. Defaults to one layer with 16 neurons.
        validation_split: Fraction of data used to validation during training. Defaults to 0.15.
        test_split: Fraction of data used to testing. Defaults to 0.15.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer.
        last_activation: Activation function used in the output layer. Defaults to 'softmax'.
        kernel_regularizer: Kernel regularizer to be used. 'l1', 'l2' or 'l1_l2'. Optional parameter.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'categorical_crossentropy'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop. Optional parameter.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['accuracy'].

    Returns:
        Trained MLP, training history and scalar test loss or list of scalars.
    """

    # 1 Create model and add layers
    model = keras.Sequential()

    regularizer = keras.regularizers.get(kernel_regularizer)

    for neuron in neurons:
        model.add(keras.layers.Dense(units=neuron, activation=activation, kernel_regularizer=regularizer))

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

    # 2 Train model and validate
    model, history, evaluation = _train_and_validate(
        X=X,
        y=y,
        model=model,
        validation_split=validation_split,
        test_split=test_split,
        output_neurons=output_neurons,
        kernel_regularizer=regularizer,
        last_activation=last_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss_function=loss_function,
        metrics=metrics,
        early_stopping=early_stopping,
        es_patience=es_patience,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history, evaluation


@beartype
def predict_MLP(
    model: keras.Sequential,
    data: np.ndarray,
) -> np.ndarray:
    """
    Use trained MLP (Multilayer Perceptron) to make predictions for data using Keras.

    Args:
        model: Trained MLP model.
        data: Data to make predictions for.

    Returns:
        Predictions for the input samples.
    """

    predictions = model.predict(data)
    return predictions


@check_keras_training_arguments
@beartype
def train_and_validate_CNN(
    X: np.ndarray,
    y: np.ndarray,
    kernel_size: Union[int, Sequence[Tuple[int, int]]],
    neurons: Sequence[int] = [16],
    validation_split: float = 0.15,
    test_split: float = 0.15,
    activation: Literal["relu"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    optimizer: Literal["adam"] = "adam",
    learning_rate: float = 0.001,
    pool_size: int = 2,
    loss_function: Literal["categorical_crossentropy", "binary_crossentropy", "mse"] = "categorical_crossentropy",
    kernel_regularizer: Optional[Literal["l1", "l2", "l1_l2"]] = None,
    dropout_rate: Optional[None] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    metrics: Optional[Sequence[Literal["accuracy", "precision", "recall"]]] = [
        "accuracy"
    ],  # NOTE: Is this a useful parameter? Should there be more options?
) -> Tuple[keras.Sequential, dict, list]:
    """
    Train and validate CNN (Convolutional Neural Network) using Keras.

    Args:
        X: Input data.
        y: Target labels.
        kernel_size: Height and width of the 2D convolution window.
        neurons: Number of neurons in each hidden layer. Defaults to one layer with 16 neurons.
        validation_split: Fraction of data used to validation during training. Defaults to 0.15.
        test_split: Fraction of data used to testing. Defaults to 0.15.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer.
        last_activation: Activation function used in the output layer. Defaults to 'softmax'.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Defalts to 0.001.
        pool size: Window size over which to take the maximum.
        loss_function: Loss function to be used. Defaults to 'categorical_crossentropy'.
        kernel_regularizer: Kernel regularizer to be used. 'l1', 'l2' or 'l1_l2'. Optional parameter.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop. Optional parameter.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        batch_size: Number of samples per gradient update. Defaults to 32.
        epochs: Number of epochs to train the model. Defaults to 50.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['accuracy'].

    Returns:
        Trained CNN, training history and scalar test loss or list of scalars.
    """

    if pool_size == 0:
        raise exceptions.InvalidParameterValueException("Pool size should be greater than 0.")

    # Step 1 Create model and add layers
    model = keras.Sequential()

    regularizer = keras.regularizers.get(kernel_regularizer)

    for neuron in neurons:
        model.add(
            keras.layers.Conv2D(
                filters=neuron,
                kernel_size=kernel_size,
                activation=activation,
                padding="same",
                kernel_regularizer=regularizer,
            )
        )

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=pool_size))

    # Step 2 Train model and validate
    model, history, evaluation = _train_and_validate(
        X=X,
        y=y,
        model=model,
        validation_split=validation_split,
        test_split=test_split,
        output_neurons=output_neurons,
        kernel_regularizer=regularizer,
        last_activation=last_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss_function=loss_function,
        metrics=metrics,
        early_stopping=early_stopping,
        es_patience=es_patience,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history, evaluation


@beartype
def predict_CNN(
    model: keras.Sequential,
    data: np.ndarray,
) -> np.ndarray:
    """
    Use trained CNN (Convolutional Neural Network) to make predictions for data using Keras.

    Args:
        model: Trained CNN model.
        data: Data to make predictions for.

    Returns:
        Predictions for the input samples.
    """
    predictions = model.predict(data)
    return predictions
