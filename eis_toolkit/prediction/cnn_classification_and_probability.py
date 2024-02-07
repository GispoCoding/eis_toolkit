from typing import Any, Literal, Union

import numpy as np
import tensorflow as tf
from beartype import beartype
from keras import Model
from numpy import long, ndarray, signedinteger
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import CNNException, CNNRunningParameterException, InvalidArgumentTypeException


@beartype
def __do_the_cnn(
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
     Do an instance of CNN. This is a private function that can be used to create an instance of CNN.

    Parameters:
       input_shape_for_cnn: shape of the input.
       convolution_kernel_size: size of the kernel (usually is the last number of the input tuple)
       pool_size: size of the pooling layer
       conv_list: list of unit for the conv layers.
       regularization: Type of regularization to insert into the CNN or MLP.
       data_augmentation: if you want data augmentation or not (Random rotation is implemented).
       optimizer: select one optimizer for the MLP.
       loss: the loss function used to calculate accuracy.
       neuron_list: List of unit or neuron used to build the network.
       dropout_rate: if you want to use dropout add a number as floating point.
       output_units: number of output classes.
       last_activation: usually you should use softmax or sigmoid.

    Return:
        return the compiled model.

    Raises:
        CNNException: raised when the input is not valid
    """

    # check that the input is not null
    if input_shape_for_cnn is None:
        raise CNNException

    if len(neuron_list) <= 0 or dropout_rate <= 0:
        raise InvalidArgumentTypeException

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

    # we flatten
    body = tf.keras.layers.Flatten()(body)

    # create the model
    classifier = tf.keras.layers.Dense(output_units, activation=last_activation, kernel_regularizer=regularization)(
        body
    )

    # create the model
    model = tf.keras.Model(inputs=input_layer, outputs=classifier)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# now let's prepare two mega function one for classification and one for regression
@beartype
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
) -> tuple[Model, list[signedinteger[Any] | long], list[signedinteger[Any] | long | Any], Any]:
    """
    Do training and evaluation of the model with cross validation.

    Parameters:
         x_train: This is the dataset you need to analyse,
         y_train: partition of the labels used for training they can be encoded (done with OHE) or a list of integers,
         x_validation: This is the partition of the dataset used for validation,
         y_validation: partition of the labels used for validation in the same form of the y_train,
         batch_size: how much we want the batch size,
         epochs: how many epochs we want to run the model,
         input_shape_for_cnn: shape of the inputs windows -> tuple[int, int, int] just a point -> tuple[int, int],
         convolutional_kernel_size: size of the kernel (usually is the last number of the input tuple)
         pool_size: size of the pooling layer
         conv_list: list of unit for the conv layers.
         sample_weights: if you want to sample weights,
         neuron_list: How deep you want to MLP
         dropout_rate: float number that help avoiding over-fitting,
         regularization: regularization of each MLP layers, None if you do not want it.
         data_augmentation: bool in case you use windows you can have random rotation,
         optimizer: loss optimization function,
         output_units: how many class you have to predicts

    Return:
        return the compiled model, the true validation labels, the predicted labels, score

    Raises:
        CNNRunningParameterException: when the batch size or epochs are wrong integer
    """

    if batch_size <= 0 or epochs <= 0:
        raise CNNRunningParameterException

    if len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise CNNRunningParameterException

    # generate the model
    cnn_model = __do_the_cnn(
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

    stacked_true, stacked_prediction = list(), list()

    _ = cnn_model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
    )

    # make the score and the prediction
    score = cnn_model.evaluate(x_validation, y_validation)[1]
    prediction = cnn_model.predict(x_validation)

    stacked_true.append(np.argmax(y_validation))
    stacked_prediction.append(np.argmax(prediction))

    # create a cm
    # cm = confusion_matrix(np.array(stacked_true), np.array(stacked_prediction), normalize="all")
    # df = pd.DataFrame(cm, columns=["Non deposit", "deposit"], index=["Non deposit", "deposit"])
    return cnn_model, stacked_true, stacked_prediction, score


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
) -> tuple[Model, ndarray, list[int], Any]:
    """
    Do training and evaluation of the model with cross validation.

    Parameters:
         x_train: This is the dataset you need to analyse,
         y_train: partition of the labels used for training they can be encoded (done with OHE) or a list of integers,
         x_validation: This is the partition of the dataset used for validation,
         y_validation: partition of the labels used for validation in the same form of the y_train,
         threshold: number that determine bound between positive or negative,
         batch_size: how much we want the batch size,
         convolutional_kernel_size: size of the kernel (usually is the last number of the input tuple)
         pool_size: size of the pooling layer
         conv_list: list of unit for the conv layers.
         epochs: how many epochs we want to run the model,
         input_shape_for_cnn: shape of the inputs windows -> tuple[int, int, int] just a point -> tuple[int, int],
         sample_weights: if you want to sample weights,
         neuron_list: How deep you want to MLP
         dropout_rate: float number that help avoiding over-fitting,
         regularization: regularization of each MLP layers, None if you do not want it.
         data_augmentation: bool in case you use windows you can have random rotation,
         optimizer: loss optimization function,
         output_units: how many class you have to predicts

    Return:
        return the compiled model, the true validation labels, the predicted labels, score

    Raises:
        CNNRunningParameterException: when the batch size or epochs are wrong integer
    """

    if batch_size <= 0 or epochs <= 0:
        raise CNNRunningParameterException

    if len(conv_list) <= 0 or len(neuron_list) <= 0:
        raise CNNRunningParameterException

    if threshold <= 0.0:
        raise CNNRunningParameterException

    # generate the model
    cnn_model = __do_the_cnn(
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

    stacked_prediction = list()

    _ = cnn_model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
    )

    # make the score and the prediction
    score = cnn_model.evaluate(x_validation, y_validation)[1]
    prediction = cnn_model.predict(x_validation)

    if prediction[0] <= threshold:
        stacked_prediction.append(0)
    else:
        stacked_prediction.append(1)

    # create a cm
    # cm = confusion_matrix(np.array(stacked_true), np.array(stacked_prediction), normalize="all")
    # df = pd.DataFrame(cm, columns=["Non deposit", "deposit"], index=["Non deposit", "deposit"])
    return cnn_model, y_validation, stacked_prediction, score
