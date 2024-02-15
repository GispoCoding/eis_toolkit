from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from beartype import beartype
from keras import Model
from numpy import ndarray
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import InvalidParameterValueException


@beartype
def _create_an_instance_of_cnn(
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int]],
    convolution_kernel_size: tuple[int, int],
    conv_list: list[int],
    neuron_list: list[int],
    pool_size: int = 2,
    dropout_rate: Union[None, float] = None,
    last_activation: Literal["softmax", "sigmoid", None] = "softmax",
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=Union[
        tf.keras.losses.BinaryCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.MeanSquaredError(),
    ],
    output_units=2,
    metrics="accuracy",
) -> tf.keras.Model:
    """
    Create an instance of the CNN model.

    Args:
        input_shape_for_cnn: Shape of the input data. It can be:
            - For windows: (h, w, c) where h is the height, w is the width, and c is the number of channels.
            - For points: (p, c) where p is the point value and c is the dimension of the point.
        convolution_kernel_size: Size of the convolution kernel.
        conv_list: List of units for the convolution layers.
        neuron_list: List of units for the fully connected layers.
        pool_size: Size of the pooling layer. Defaults to 2.
        dropout_rate: Dropout rate to prevent overfitting. Defaults to None.
        last_activation: Activation function for the last layer. Defaults to "softmax".
        regularization: Regularization for the layers. Defaults to None.
        data_augmentation: Enable data augmentation. Defaults to False.
        optimizer: Optimizer for training. Defaults to "Adam".
        loss: Loss function for training. Defaults to BinaryCrossentropy.
        output_units: Number of output units. Defaults to 2.
        metrics: Evaluation metric for training. Defaults to "accuracy".

    Returns:
        Compiled Keras Model.

    Raises:
        InvalidParameterValueException: Raised when the input parameters are invalid.
    """

    # check that the input is not null
    if input_shape_for_cnn is None:
        raise InvalidParameterValueException

    if len(conv_list) <= 0:
        raise InvalidParameterValueException

    if len(neuron_list) <= 0:
        raise InvalidParameterValueException

    if dropout_rate is not None and dropout_rate <= 0:
        raise InvalidParameterValueException

    # generate the input
    input_layer = tf.keras.Input(shape=input_shape_for_cnn)

    if data_augmentation is not False:
        input_layer = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

    # we do dynamically the building of the conv2d layers
    for conv_layer_number, neuron in enumerate(conv_list):
        if conv_layer_number == 0:
            x = tf.keras.layers.Conv2D(
                filters=neuron,
                activation="relu",
                padding="same",
                kernel_regularizer=regularization,
                kernel_size=convolution_kernel_size,
            )(input_layer)
        else:
            x = tf.keras.layers.Conv2D(
                filters=neuron,
                activation="relu",
                padding="same",
                kernel_regularizer=regularization,
                kernel_size=convolution_kernel_size,
            )(x)

        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)

    for layer_number, neuron in enumerate(neuron_list):
        if layer_number == 0:
            body = tf.keras.layers.Dense(neuron, activation="relu", kernel_regularizer=regularization)(input_layer)
        else:
            body = tf.keras.layers.Dense(neuron, kernel_regularizer=regularization, activation="relu")(body)

        if dropout_rate is not None:
            body = tf.keras.layers.Dropout(dropout_rate)(body)

    body = tf.keras.layers.Flatten()(body)

    classifier = tf.keras.layers.Dense(output_units, activation=last_activation, kernel_regularizer=regularization)(
        body
    )

    model = tf.keras.Model(inputs=input_layer, outputs=classifier)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


def run_inference(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    epochs: int,
    conv_list: list[int],
    neuron_list: list[int],
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int], tuple[int], int],
    convolutional_kernel_size: tuple[int, int],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pool_size: int = 2,
    sample_weights: bool = False,
    dropout_rate: Union[None, float] = None,
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    output_units=2,
    last_activation_layer: Literal["softmax", "sigmoid", None] = "softmax",
    loss_function: Union[
        tf.keras.losses.BinaryCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.MeanAbsoluteError(),
    ] = tf.keras.losses.BinaryCrossentropy(),
) -> tuple[Model, ndarray or None, Any or None]:
    """
    Train and evaluate a Convolutional Neural Network (CNN) for data classification.

    This function allows for flexible CNN architecture specification, including the number of convolutional
    and dense layers, input shape, and various hyperparameters. It supports data augmentation, dropout, and
    regularization to prevent overfitting. The function can work with both raw data points and structured
    input like images or sequences.

    Args:
        X: Input data for the model.
        y: Target labels for the input data. They can be encoded using either one-hot encoding (OHE) or represented as a list of integers.
        batch_size: Number of samples per gradient update.
        epochs: Number of epochs to train the model.
        conv_list: Number of filters in each convolutional layer.
        neuron_list: Number of neurons in each dense layer.
        input_shape_for_cnn: Shape of the input data.
        convolutional_kernel_size: Size of the convolution kernels.
        validation_split: Fraction of the data to use as validation set. Defaults to 0.2.
        validation_data: Explicit validation set.
        pool_size: Size of the pooling windows. Defaults to 2.
        sample_weights: Whether to use sample weighting. Defaults to False.
        dropout_rate: Fraction of the input units to drop. Defaults to None.
        regularization: Regularization function applied to the activation functions. Defaults to None.
        data_augmentation: Whether to use data augmentation. Defaults to False.
        optimizer: Optimization algorithm. Defaults to "Adam".
        output_units: Number of output units in the final layer. Defaults to 2.
        last_activation_layer: Activation function for the output layer. Defaults to "softmax".
        loss_function: Loss function for the optimization. Defaults to `tf.keras.losses.BinaryCrossentropy()`.

    Returns:
        A tuple containing the trained model, predictions on the validation set, and evaluation score.

    Raises:
        InvalidParameterValueException: If any input parameter is invalid.
    """

    # Validation checks for input parameters
    if X.size == 0 or y.size == 0 or batch_size <= 0 or epochs <= 0 or len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise InvalidParameterValueException("Input parameters have invalid values.")

    if dropout_rate is not None and (dropout_rate <= 0 or dropout_rate > 1):
        raise InvalidParameterValueException("Dropout rate must be in the range (0, 1].")

    # Model instantiation
    cnn_model = _create_an_instance_of_cnn(
        input_shape_for_cnn=input_shape_for_cnn,
        convolution_kernel_size=convolutional_kernel_size,
        conv_list=conv_list,
        pool_size=pool_size,
        neuron_list=neuron_list,
        dropout_rate=dropout_rate,
        last_activation=last_activation_layer,
        regularization=regularization,
        data_augmentation=data_augmentation,
        optimizer=optimizer,
        loss=loss_function,
        output_units=output_units,
    )

    # Model training
    _ = cnn_model.fit(
        X,
        y,
        validation_split=validation_split if validation_split is not None else None,
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y) if sample_weights is not False else None,
    )
    
    # Evaluation on validation data if provided
    if validation_data is not None:
        x_valid, y_valid = validation_data
        score = cnn_model.evaluate(x_valid, y_valid)[1]
        prediction = cnn_model.predict(x_valid)

        return cnn_model, prediction, score
    
    return cnn_model, None, None
