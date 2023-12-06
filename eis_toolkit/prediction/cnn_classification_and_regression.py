from typing import Literal, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
from beartype import beartype
from keras import Model
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import CNNException, CNNRunningParameterException, InvalidArgumentTypeException
from eis_toolkit.prediction.model_performance_estimation import performance_model_estimation


def normalize_the_data(scaler_agent: sklearn.preprocessing, data: np.ndarray):
    """
    Do Data normalization.

    Parameters:
       scaler_agent: this is the scaler agent used for data normalization is like an handler.
       data: data to normalize

    Return:
        return normalized data.
    """
    number_of_sample, h, w, c = data.shape
    temp = scaler_agent.transform(data.reshape(-1, data.shape[-1]))
    normalized_data = temp.reshape(number_of_sample, h, w, c)
    return normalized_data


@beartype
def make_one_hot_encoding(labels: np.ndarray):
    """
     Do the OneHotEncoding.

    Parameters:
       labels: labels to encode.

    Return:
        return encoded labels.

    Raises:
        InvalidArgumentTypeException: labels are None.
    """
    if labels is None:
        raise InvalidArgumentTypeException

    # to categorical
    enc = OneHotEncoder(handle_unknown="ignore")
    # train and valid set
    temp = np.reshape(labels, (-1, 1))
    label_encoded = enc.fit_transform(temp).toarray()
    return label_encoded


@beartype
def do_the_cnn(
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int]],
    convolution_kernel_size: tuple[int, int],
    conv_list: list[int] = [8, 16],
    pool_size: int = 2,
    neuron_list: list[int] = [16],
    dropout_rate: Union[None, float] = None,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=Union[tf.keras.losses.BinaryCrossentropy, tf.keras.losses.CategoricalCrossentropy],
    output_units=2,
) -> tf.keras.Model:
    """
     Do an instance of CNN.

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
        InvalidArgumentTypeException: one of the arguments is invalid.
        CNNException: raised when the input is not valid
    """
    # if regression and binary we can not uses more than 1
    if output_units > 1 and loss == tf.keras.losses.BinaryCrossentropy:
        raise InvalidArgumentTypeException

    # check that the input is not null
    if input_shape_for_cnn is None:
        raise CNNException

    if len(neuron_list) <= 0 or dropout_rate <= 0:
        raise InvalidArgumentTypeException

    # generate the input
    input_layer = tf.keras.Input(shape=input_shape_for_cnn)

    if data_augmentation is not False:
        input_layer = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

    # we do dynamically the conv2d
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
    classifier = tf.keras.layers.Dense(
        output_units, activation=last_activation, kernel_regularizer=regularization, name="classifier"
    )(body)

    # create the model
    model = tf.keras.Model(inputs=input_layer, outputs=classifier, name="the_mlp_model")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# now let's prepare two mega function one for classification and one for regression
@beartype
def train_and_predict_for_classification(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    epochs: int,
    cross_validation: Literal["LOOCV", "KFOLD", "SKFOLD"],
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int], tuple[int], int],
    convolutional_kernel_size: tuple[int, int],
    conv_list: list[int] = [8, 16],
    pool_size: int = 2,
    sample_weights: bool = False,
    neuron_list: list[int] = [16],
    dropout_rate: Union[None, float] = None,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    output_units=2,
) -> tuple[Model, DataFrame]:
    """
    Do training and evaluation of the model with cross validation.

    Parameters:
         X: This is the dataset,
         y: labels,
         batch_size: how much we want the batch size,
         epochs: how many epochs we want to run the model,
         cross_validation: Type of cross validation
         input_shape_for_cnn: shape of the inputs windows -> tuple[int, int, int] just a point -> tuple[int, int],
         convolutional_kernel_size: size of the kernel (usually is the last number of the input tuple)
         pool_size: size of the pooling layer
         conv_list: list of unit for the conv layers.
         sample_weights: if you want to samples weights,
         neuron_list: How deep you want to MLP
         dropout_rate: float number that help avoiding overfitting,
         last_activation: type of last activation I suggest here to use softmax,
         regularization: regularization of each MLP layers, None if you do not want it.
         data_augmentation: bool in case you use windows you can have random rotation,
         optimizer: loss optimization function,
         loss: loss functyion I suggest this -> tf.keras.losses.CategoricalCrossentropy,
         output_units: how many class you have to predicts

    Return:
        return pd dataframe that contains the confusion matrix and instance of the best model.

    Raises:
        CNNRunningParameterException: when the batch size or epochs are wrong integer
        InvalidArgumentTypeException: when you try to use sigmoid or BinaryCrossEntropy for classification.
    """

    if batch_size <= 0 or epochs <= 0:
        raise CNNRunningParameterException

    if last_activation == "sigmoid" or loss == tf.keras.losses.BinaryCrossentropy():
        raise InvalidArgumentTypeException

    # seems is classy we need one hot
    y_encoded = make_one_hot_encoding(labels=y)

    # generate the model
    # generate the model
    cnn_model = do_the_cnn(
        input_shape_for_cnn=input_shape_for_cnn,
        convolution_kernel_size=convolutional_kernel_size,
        conv_list=conv_list,
        pool_size=pool_size,
        neuron_list=neuron_list,
        dropout_rate=dropout_rate,
        last_activation=last_activation,
        regularization=regularization,
        data_augmentation=data_augmentation,
        optimizer=optimizer,
        loss=loss,
        output_units=output_units,
    )

    # prepare the scaler
    # get cross validation methods
    selected_cs = performance_model_estimation(cross_validation_type=cross_validation, number_of_split=1)

    stacked_true, stacked_prediction = list(), list()
    best_score = 0
    model_to_return = None

    scaler_agent = StandardScaler()
    scaler_agent.fit(X.reshape(-1, X.shape[-1]))

    for i, (train_index, test_index) in enumerate(selected_cs.split(y)):
        # train test
        X_train = normalize_the_data(scaler_agent=scaler_agent, data=X[train_index])
        y_train = y_encoded[train_index]

        # valid test
        X_validation = normalize_the_data(scaler_agent=scaler_agent, data=X[test_index])
        y_validation = y_encoded[test_index]

        _ = cnn_model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
        )

        # make the score and the prediction
        score = cnn_model.evaluate(X_validation, y_validation)[0]
        prediction = cnn_model.predict(X_validation)

        stacked_true.append(np.argmax(y_validation))
        stacked_prediction.append(np.argmax(prediction))

        if score > best_score:
            best_score = score
            model_to_return = cnn_model

    # create a cm
    cm = confusion_matrix(np.array(stacked_true), np.array(stacked_prediction), normalize="all")
    df = pd.DataFrame(cm, columns=["Non deposit", "deposit"], index=["Non deposit", "deposit"])
    return model_to_return, df


@beartype
def train_and_predict_for_regression(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    epochs: int,
    threshold: float,
    cross_validation: Literal["LOOCV", "KFOLD", "SKFOLD"],
    input_shape_for_cnn: Union[tuple[int, int, int], tuple[int, int], tuple[int], int],
    convolutional_kernel_size: tuple[int, int],
    sample_weights: bool = False,
    conv_list: list[int] = [8, 16],
    neuron_list: list[int] = [16],
    pool_size: int = 2,
    dropout_rate: Union[None, float] = None,
    last_activation: Literal["softmax", "sigmoid"] = "sigmoid",
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    output_units=1,
) -> tuple[Model, DataFrame]:
    """
    Do training and evaluation of the model with cross validation.

    Parameters:
         X: This is the dataset,
         y: labels,
         threshold: number that determine bound between positive or negative,
         batch_size: how much we want the batch size,
         convolutional_kernel_size: size of the kernel (usually is the last number of the input tuple)
         pool_size: size of the pooling layer
         conv_list: list of unit for the conv layers.
         epochs: how many epochs we want to run the model,
         cross_validation: Type of cross validation
         input_shape_for_cnn: shape of the inputs windows -> tuple[int, int, int] just a point -> tuple[int, int],
         sample_weights: if you want to sample weights,
         neuron_list: How deep you want to MLP
         dropout_rate: float number that help avoiding overfitting,
         last_activation: type of last activation I suggest here to use softmax,
         regularization: regularization of each MLP layers, None if you do not want it.
         data_augmentation: bool in case you use windows you can have random rotation,
         optimizer: loss optimization function,
         loss: loss function I suggest this -> tf.keras.losses.CategoricalCrossentropy,
         output_units: how many class you have to predicts

    Return:
        return pd dataframe that contains the confusion matrix and instance of the best model.

    Raises:
        CNNRunningParameterException: when the batch size or epochs are wrong integer
        InvalidArgumentTypeException: when you try to use softmax or CategoricalCrossEntropy for regression.
    """

    if batch_size <= 0 or epochs <= 0:
        raise CNNRunningParameterException

    if last_activation == "softmax" or loss == tf.keras.losses.CategoricalCrossentropy():
        raise InvalidArgumentTypeException

    # generate the model
    cnn_model = do_the_cnn(
        input_shape_for_cnn=input_shape_for_cnn,
        convolution_kernel_size=convolutional_kernel_size,
        conv_list=conv_list,
        pool_size=pool_size,
        neuron_list=neuron_list,
        dropout_rate=dropout_rate,
        last_activation=last_activation,
        regularization=regularization,
        data_augmentation=data_augmentation,
        optimizer=optimizer,
        loss=loss,
        output_units=output_units,
    )

    # prepare the scaler
    scaler_agent = StandardScaler()
    scaler_agent.fit(X.reshape(-1, X.shape[-1]))

    # get cross validation methods
    selected_cs = performance_model_estimation(cross_validation_type=cross_validation, number_of_split=1)

    stacked_true, stacked_prediction = list(), list()
    best_score = 0
    model_to_return = None

    for i, (train_index, test_index) in enumerate(selected_cs.split(y)):
        # train test
        X_train = normalize_the_data(scaler_agent=scaler_agent, data=X[train_index])
        y_train = y[train_index]

        # valid test
        X_validation = normalize_the_data(scaler_agent=scaler_agent, data=X[test_index])
        y_validation = y[test_index]

        _ = cnn_model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=compute_sample_weight("balanced", y_train) if sample_weights is not False else None,
        )

        # make the score and the prediction
        score = cnn_model.evaluate(X_validation, y_validation)[0]
        prediction = cnn_model.predict(X_validation)

        stacked_true.append(y_validation)

        if prediction[0] <= threshold:
            stacked_prediction.append(0)
        else:
            stacked_prediction.append(1)

        if score > best_score:
            best_score = score
            model_to_return = cnn_model

    # create a cm
    cm = confusion_matrix(np.array(stacked_true), np.array(stacked_prediction), normalize="all")
    df = pd.DataFrame(cm, columns=["Non deposit", "deposit"], index=["Non deposit", "deposit"])
    return model_to_return, df
