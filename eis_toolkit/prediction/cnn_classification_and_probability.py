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
       metrics: In case of classification accuracy is the best metrics, otherwise for regression MAE is the way.

    Return:
         the model that has been created.

    Raises:
        InvalidParameterValueException: Raised when the input shape of the CNN is not valid. It can be risen.
                                        For example, when the user plan to build a windows CNN approach and feed it
                                        with point. Moreover, it is raised when argument of the function is invalid.
                                        It is applied to convolution layers and dense layers.

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
    Do a CNN for training and evaluation of data. It is designed to classify the data provided in the given labels.

    Here, you can select how many convolutional layer and Dense layer you  want. This CNN can be used to classify
    the data provided as classes. The CNN works both with windows and points. If you select windows, iti is possible
    to use random rotation to augment the data. In case the CNN is over-fitting, you can try to avoid it adding
    dropout or regularization.

    Args:
         X: This is the dataset used for the model inference.
         y: The labels used for training: they can be encoded (done with OHE) or a list of integers.
         validation_split: split between train and validation of the data.
         validation_data: Partition of dataset used as validation (unseen data).
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
         last_activation_layer: How the output of the network is calculated.
         loss_function: The loss function used to measure how good the CNN is performing.

    Returns:
        The trained model together with the predicted values and the model's score.  When
        the validation set is None, true labels, predicted labels and scores assumes None.
    Raises:
        InvalidParameterValueException: Raised when argument of the function is invalid. It is applied to convolution
                                        layers and dense layers. Moreover, Raised when the input shape of the CNN is not
                                        valid. It can be risen. For example, when the user plan to build a windows CNN
                                        approach and feed it with point.
    """

    if X.size == 0 or y.size == 0:
        raise InvalidParameterValueException

    if batch_size <= 0 or epochs <= 0:
        raise InvalidParameterValueException

    if len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise InvalidParameterValueException

    if dropout_rate is not None and dropout_rate <= 0:
        raise InvalidParameterValueException

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

    _ = cnn_model.fit(
        X,
        y,
        validation_split=validation_split if validation_split is not None else None,
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y) if sample_weights is not False else None,
    )

    if validation_data is not None:
        x_valid, y_valid = validation_data
        score = cnn_model.evaluate(x_valid, y_valid)[1]
        prediction = cnn_model.predict(x_valid)

        return cnn_model, prediction, score
    else:
        return cnn_model, None, None
