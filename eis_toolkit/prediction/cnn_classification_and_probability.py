from typing import Any, Literal, Union

import numpy as np
import tensorflow as tf
from beartype import beartype
from keras import Model
from numpy import ndarray
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import InvalidArgumentTypeException, InvalidParameterValueException


@beartype
def _create_an_instance_of_cnn(
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int]],
    convolution_kernel_size: tuple[int, int],
    conv_list: list[int],
    neuron_list: list[int],
    pool_size: int = 2,
    dropout_rate: Union[None, float] = None,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=Union[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()],
    output_units=2,
) -> tf.keras.Model:
    """
     Do an instance of the CNN. Just the body of the CNN.

    Args::
       input_shape_for_cnn: Shape of the input. How big the input should be. It is possible to build an input:
                         - windows: (h, w, c) windows height, windows width and channel or windows dimension.
                         - point (p, c) point value and dimension of the point.
       convolution_kernel_size: Size of the kernel (usually is the last number of the input tuple). Usually is the "c"
                                of the input_shape_for_cnn.
       pool_size: Size of the pooling layer (How much you want to reduce the dimension of the data).
       conv_list: list of unit for the conv layers. Here the length of the list is the number of convolution layers.
       regularization: Regularization of each  layer. None if you do not want it:
                     - None if you prefer to avoid regularization.
                     - L1 for L1 regularization.
                     - L2 for L2 regularization.
                     - L1L2 for L1 L2 regularization.
       data_augmentation: If you want data augmentation or not (Random rotation is implemented).
       optimizer: Select one optimizer for the CNN. The default value is Adam.
       loss: The loss function used to measure how good the CNN is performing.
       neuron_list: List of unit or neuron used to build the network final part of the network. the length of the list
                    is the number of fully connected layers.
       dropout_rate: Float number that help avoiding over-fitting. It just randomly drops samples.
       output_units: number of output classes.
       last_activation: usually you should use softmax or sigmoid.

    Return:
        model: the model that has been created.

    Raises:
        CNNException: raised when the input is not valid
    """

    # check that the input is not null
    if input_shape_for_cnn is None:
        raise InvalidParameterValueException

    if len(neuron_list) <= 0:
        raise InvalidArgumentTypeException

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
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def train_and_predict_for_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int,
    epochs: int,
    conv_list: list[int],
    neuron_list: list[int],
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int], tuple[int], int],
    convolutional_kernel_size: tuple[int, int],
    pool_size: int = 2,
    sample_weights: bool = False,
    dropout_rate: Union[None, float] = None,
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    output_units=2,
) -> tuple[Model, ndarray, ndarray, Any]:
    """
    Do a CNN for training and evaluation of data. It is designed to classify the data provided in the given labels.

    Here, you can select how many convolutional layer and Dense layer you  want. This CNN can be used to classify
    the data provided as classes. The CNN works both with windows and points. If you select windows, iti is possible
    to use random rotation to augment the data. In case the CNN is over-fitting, you can try to avoid it adding
    dropout or regularization.

    Args:
         x_train: This is the portion of the data used for training the model.
         y_train: Partition of the labels used for training: they can be encoded (done with OHE) or a list of integers.
         x_validation: This is the partition of the dataset used for validation.
         y_validation: Partition of the labels used for validation in the same form of the y_train.
         batch_size: How much we want the batch size. This is the number os samples that the CNN takes during each
                     iteration.
         epochs: How many iterations we want to run the model.
         input_shape_for_cnn: Shape of the inputs windows should follow:
                              - tuple[int, int, int] feed the cnn with windows: format (h, w, c).
                              - tuple[int, int] feed a network with points: format (value, c).
         convolutional_kernel_size: Size of the kernel (usually is the last number of the input tuple).
         pool_size: Size of the pooling layer (How much you want to reduce the dimension of the data).
         conv_list: List of units used in each convolutional layer.The length of the list is the number of layers.
         sample_weights: If you want to sample weights. It is used when there is strong imbalance of the data.
                  neuron_list: This is a list of dense layers used for the last section of the network (fully connected
                  layers). The length of the list shows the number of layers.
         neuron_list: This is a list of dense layers used for the last section of the network (fully connected layers).
                      The length of the list shows the number of layers.
         dropout_rate: Float number that help avoiding over-fitting. It just randomly drops samples.
         regularization: Regularization of each  layer. None if you do not want it:
                         - None if you prefer to avoid regularization.
                         - L1 for L1 regularization.
                         - L2 for L2 regularization.
                         - L1L2 for L1 L2 regularization.
         data_augmentation: Usable only if the network is fed by windows.
         optimizer: Loss optimization function, default is Adam.
         output_units: How many class you have to predicts.

    Return:
        cnn_model: The compiled model used in this instance.
        true_value_of_the_labels: The true labels values of the data. (the test section).
        predicted_values: The predicted labels of the data. (the test section).
        score: Value of the evaluation of the network.

    Raises:
        InvalidArgumentTypeException: Argument like train adapter, valid adapter are invalid.
        InvalidParameterValueException: Argument like convolution list, neuron list are invalid.
    """

    true_value_of_the_labels, predicted_values = list(), list()

    if x_train.size == 0 or y_train.size == 0 or y_validation.size == 0 or x_validation.size == 0:
        raise InvalidArgumentTypeException

    if batch_size <= 0 or epochs <= 0:
        raise InvalidArgumentTypeException

    if len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise InvalidArgumentTypeException

    if dropout_rate is not None and dropout_rate <= 0:
        raise InvalidParameterValueException

    cnn_model = _create_an_instance_of_cnn(
        input_shape_for_cnn=input_shape_for_cnn,
        convolution_kernel_size=convolutional_kernel_size,
        conv_list=conv_list,
        pool_size=pool_size,
        neuron_list=neuron_list,
        dropout_rate=dropout_rate,
        last_activation="softmax",
        regularization=regularization,
        data_augmentation=data_augmentation,
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        output_units=output_units,
    )

    _ = cnn_model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
    )

    score = cnn_model.evaluate(x_validation, y_validation)[1]
    prediction = cnn_model.predict(x_validation)

    true_value_of_the_labels.append(np.argmax(y_validation))
    predicted_values.append(np.argmax(prediction))

    return cnn_model, y_validation, prediction, score


@beartype
def train_and_predict_for_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int,
    epochs: int,
    threshold: float,
    conv_list: list[int],
    neuron_list: list[int],
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int], tuple[int], int],
    convolutional_kernel_size: tuple[int, int],
    sample_weights: bool = False,
    pool_size: int = 2,
    dropout_rate: Union[None, float] = None,
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    output_units=1,
) -> tuple[Model, ndarray, ndarray, ndarray, Any]:
    """
    Do a CNN for training and evaluation of data. It is designed to classify pixels using threshold.

    Here, you can select how many convolutional layer and Dense layer you  want.  This CNN can be used to classify
    the data provided as classes. In addition, this function uses threshold as boundary lines between classes.
    The CNN works both with windows and points. If you select windows,  it is possible to use random rotation to augment
    the data. In case the  CNN is over-fitting, you can try to avoid  it adding dropout or regularization.

    Args:
         x_train: This is the portion of the data used for training the model.
         y_train: Partition of the labels used for training: they can be encoded (done with OHE) or a list of integers.
         x_validation: This is the partition of the dataset used for validation.
         y_validation: Partition of the labels used for validation in the same form of the y_train.
         batch_size: How much we want the batch size. This is the number os samples that the CNN takes during each
                     iteration.
         epochs: How many iterations we want to run the model.
         input_shape_for_cnn: Shape of the inputs windows should follow:
                              - tuple[int, int, int] feed the cnn with windows: format (h, w, c).
                              - tuple[int, int] feed a network with points: format (value, c).
         convolutional_kernel_size: Size of the kernel (usually is the last number of the input tuple).
         conv_list: list of units used in each convolutional layer.The length of the list is the number of layers.
         epochs: how many epochs we want to run the model,
         input_shape_for_cnn: shape of the inputs windows -> tuple[int, int, int] just a point -> tuple[int, int],
         sample_weights: If you want to sample weights. It is used when there is strong imbalance of the data.
         neuron_list: This is a list of dense layers used for the last section of the network (fully connected layers).
                      The length of the list shows the number of layers.
         pool_size: Size of the pooling layer (How much you want to reduce the dimension of the data).
         dropout_rate: Float number that help avoiding over-fitting. It just randomly drops samples.
         regularization: Regularization of each  layer. None if you do not want it:
                         - None if you prefer to avoid regularization.
                         - L1 for L1 regularization.
                         - L2 for L2 regularization.
                         - L1L2 for L1 L2 regularization.
         data_augmentation: Usable only if the network is fed by windows.
         optimizer: Loss optimization function, default is Adam.
         output_units: How many class you have to predicts.
         threshold: This number is used as borderline between classes.

    Return:
        cnn_model: The compiled model used in this instance.
        true_value_of_the_labels: The true labels values of the data. (the test section).
        predicted_values: The predicted labels of the data. (the test section).
        prediction: The value of the prediction (probabilities).
        score: Value of the evaluation of the network.

    Raises:
        InvalidArgumentTypeException: Argument like train adapter, valid adapter are invalid.
        InvalidParameterValueException: Argument like convolution list, neuron list are invalid.
    """

    if x_train.size == 0 or y_train.size == 0 or y_validation.size == 0 or x_validation.size == 0:
        raise InvalidArgumentTypeException

    if batch_size <= 0 or epochs <= 0:
        raise InvalidParameterValueException

    if len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise InvalidArgumentTypeException

    if threshold <= 0.0:
        raise InvalidArgumentTypeException

    cnn_model = _create_an_instance_of_cnn(
        input_shape_for_cnn=input_shape_for_cnn,
        convolution_kernel_size=convolutional_kernel_size,
        conv_list=conv_list,
        pool_size=pool_size,
        neuron_list=neuron_list,
        dropout_rate=dropout_rate,
        last_activation="sigmoid",
        regularization=regularization,
        data_augmentation=data_augmentation,
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        output_units=output_units,
    )

    predicted_values = list()

    _ = cnn_model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
    )

    score = cnn_model.evaluate(x_validation, y_validation)[1]
    prediction = cnn_model.predict(x_validation)

    for p in prediction:
        if p <= threshold:
            predicted_values.append(0)
        else:
            predicted_values.append(1)
    return cnn_model, y_validation, np.array(predicted_values), prediction, score
