import os
from typing import Any, Literal, Union

import joblib
import numpy
import numpy as np
import tensorflow as tf
from beartype import beartype
from numpy import dtype, ndarray
from sklearn.preprocessing import StandardScaler


@beartype
def return_windows_as_sequence(
    input_path: str = None,
) -> tuple[dict[str, ndarray[Any, dtype[Any]]], ndarray[Any, dtype[Any]]]:
    """
    Load sets of windows from its folder.

    Parameters:
        input_path: this is what in keras is called optimizer.

    Return:
        It returns dataset and labels

    Raises:
        TODO
    """
    labels = list()
    data_dictionary = dict()
    for folder in os.listdir(f"{input_path}"):
        concatenated = None
        for satellites_folder in os.listdir(f"{input_path}/{folder}"):
            full_path = f"{input_path}/{folder}/{satellites_folder}"
            # loop in all the folders and get all wins
            print(f"[WALK] Walking into {full_path}")
            # rest the win list
            windows_holder = list()
            for windows in os.listdir(full_path):
                # get the class
                labels.append(windows.split(".")[0].split("_")[-1])
                windows_holder.append(np.load(f"{full_path}/{windows}"))
            # make concatenation
            if concatenated is None:
                concatenated = np.array(windows_holder)
            else:
                concatenated = np.concatenate((concatenated, np.array(windows_holder)), axis=-1)

        # add the data dict
        data_dictionary[f"{folder}"] = concatenated.astype("float32")
    # prepare the labels
    labels = np.array(labels).astype("uint8")

    return data_dictionary, labels


@beartype
def create_the_scaler(data_dictionary: dict, dump: bool = False):
    """
    Create scaler for data normalization.

    Parameters:
        data_dictionary: dictionary containing type of data and data.
        dump: if you want to dump file in a folders.

    Return:
        normalized data dictionary

    Raises:
        TODO
    """
    if dump and os.path.exists("scaler"):
        os.makedirs("scaler")

    # normalize the data
    dictionary_of_scaler = dict()
    for data in data_dictionary.keys():
        scaler = StandardScaler()
        scaler.fit(data_dictionary[data].reshape(-1, data_dictionary[data].shape[-1]))
        dictionary_of_scaler[data] = scaler

        if dump:
            joblib.dump(value=scaler, filename=f"scaler/scaler_{data}.bin")

    return dictionary_of_scaler


@beartype
def normalize_the_data(data_to_normalize, normalizator) -> np.ndarray:
    """
    Normalize the data.

    Parameters:
        data_to_normalize: dictionary containing data that need to be normalized.
        normalizator: scaler needed for data normalization.

    Return:
        normalized dataset

    Raises:
        TODO
    """
    try:
        number_of_samples, h, w, c = data_to_normalize.shape
        temp = data_to_normalize.reshape(-1, data_to_normalize.shape[-1])
        normalized_input = normalizator.transform(temp)
        return normalized_input.reshape(number_of_samples, h, w, c)
    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception {ex}")


@beartype
def convolutional_body_of_the_cnn(
    input_layer: tf.keras.Input,
    neuron_list: Union[int],
    kernel_size: tuple[int, int],
    kernel_regularizes: tf.keras.regularizers,
    pool_size: int,
    dropout: float = None,
) -> tf.keras.layers:
    """
    Do create hidden layer (cov + dropout batch norm and max pooling).

    Parameters:
        input_layer: The input layer of the network.
        neuron_list: The list of neurons to assign to each layer.
        kernel_size: The size of the kernel,
        kernel_regularizes: The type of kernel regularize.
        pool_size: how big you want the pool size,
        dropout: add the dropout layer to the body.

    Return:
        return the block of hidden layer

    Raises:
        TODO
    """
    # we do dynamically the conv2d
    for layer_number, neuron in enumerate(neuron_list):
        if layer_number == 0:
            x = tf.keras.layers.Conv2D(
                filters=neuron,
                activation="relu",
                padding="same",
                kernel_regularizer=kernel_regularizes,
                kernel_size=kernel_size,
            )(input_layer)
        else:
            x = tf.keras.layers.Conv2D(
                filters=neuron,
                activation="relu",
                padding="same",
                kernel_regularizer=kernel_regularizes,
                kernel_size=kernel_size,
            )(x)

        if dropout is not None:
            x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)

    # we flatten
    x = tf.keras.layers.Flatten()(x)
    return x


def dense_nodes(input_layer: tf.keras.Input, neuron_list: Union[int], dropout: float = None) -> tf.keras.layers:
    """
    Do the creation of dense layer for MLP.

    Parameters:
        input_layer: The input layer of the network.
        neuron_list: The list of neurons to assign to each layer.
        dropout: add the dropout layer to the body.

    Return:
        return the block of dense layer

    Raises:
        TODO
    """

    for layer_number, neuron in enumerate(neuron_list):
        if layer_number == 0:
            x = tf.keras.layers.Dense(neuron, activation="relu")(input_layer)
        else:
            x = tf.keras.layers.Dense(neuron, activation="relu")(x)

        if dropout is not None:
            x = tf.keras.layers.Dropout(dropout)(x)

    # we flatten
    x = tf.keras.layers.Flatten()(x)
    return x


@beartype
def create_multi_modal_cnn(
    input_aem: tuple[int, int, int] or tuple[int, int] = None,
    kernel_aem: tuple[int, int] = None,
    input_gravity: tuple[int, int, int] or tuple[int, int] = None,
    kernel_gravity: tuple[int, int] = None,
    input_magnetic: tuple[int, int, int] or tuple[int, int] = None,
    kernel_magnetic: tuple[int, int] = None,
    input_radiometric: tuple[int, int, int] or tuple[int, int] = None,
    kernel_radiometric: tuple[int, int] = None,
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2] = None,
    data_augmentation: bool = False,
    optimizer: str = "Adam",
    loss=Union[tf.keras.losses.BinaryCrossentropy, tf.keras.losses.CategoricalCrossentropy],
    inputs: int = 1,
    neuron_list: Union[int] = [16],
    pool_size: int = 2,
    dropout_rate: Literal[None, float] = None,
    is_a_cnn: bool = True,
    output: int = 2,
    last_activation: Literal["softmax", "sigmoid"] = "softmax",
):
    """
     Do an instance of CNN or MLP.

    Parameters:
       input_aem: if exist, shape of the input aem.
       kernel_aem: if exist channel size of the input (usually is last shape value).
       input_gravity: if exist, shape of the input gravity.
       kernel_gravity: if exist channel size of the input (usually is last shape value).
       input_magnetic: if exist, shape of the input magnetic.
       kernel_magnetic: if exist channel size of the input (usually is last shape value).
       input_radiometric: if exist, shape of the input radiometric.
       kernel_radiometric: if exist channel size of the input (usually is last shape value).
       regularization: Type of regularization to insert into the CNN or MLP.
       data_augmentation: if you want data augmentation or not (Random rotation is implemented).
       optimizer: select one optimizer for the CNN or MLP.
       loss: the loss function used to calculate accuracy.
       inputs: number of inout to assign to the CNN or MLP (1 uni-modal > 1 fusion).
       neuron_list: List of unit or neuron used to build the network.
       pool_size: the pool size used by the CNN (Max-pooling).
       dropout_rate: if you want to use dropout add a number as floating point.
       is_a_cnn: true if you want to build a CNN false if you want to build a MLP.
       output: number of output classes.
       last_activation: usually you should use softmax or sigmoid.

    Return:
        return the compiled model.

    Raises:
        TODO
    """
    if input_aem is not None:
        input_layer = tf.keras.Input(shape=input_aem, name="AEM")
        kernel = kernel_aem
    elif input_gravity is not None:
        input_layer = tf.keras.Input(shape=input_gravity, name="Gravity")
        kernel = kernel_gravity
    elif input_magnetic is not None:
        input_layer = tf.keras.Input(shape=input_magnetic, name="Magnetic")
        kernel = kernel_magnetic
    else:
        input_layer = tf.keras.Input(shape=input_radiometric, name="Radiometric")
        kernel = kernel_radiometric

    if inputs == 1:
        if data_augmentation:
            input_layer = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

        if is_a_cnn:
            body = convolutional_body_of_the_cnn(
                input_layer=input_layer,
                neuron_list=neuron_list,
                kernel_size=kernel,
                dropout=dropout_rate,
                kernel_regularizes=regularization,
                pool_size=pool_size,
            )
        else:
            body = dense_nodes(input_layer=input_layer, neuron_list=neuron_list, dropout=dropout_rate)

        # create the classy
        classifier = tf.keras.layers.Dense(output, activation=last_activation, name="classifier")(body)
        model = tf.keras.Model(inputs=input_layer, outputs=classifier, name="model_with_1_input")

        return model
    else:
        print("[NN FACTORY] Multiples input")
        model_input, model_output = list(), list()

        if input_aem is not None:
            aem = tf.keras.Input(shape=input_aem, name="AEM")

            if data_augmentation:
                aem = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

            if is_a_cnn:
                body_aem = convolutional_body_of_the_cnn(
                    input_layer=aem,
                    neuron_list=neuron_list,
                    kernel_size=kernel_aem,
                    dropout=dropout_rate,
                    kernel_regularizes=regularization,
                    pool_size=pool_size,
                )
            else:
                body_aem = dense_nodes(input_layer=aem, neuron_list=neuron_list, dropout=dropout_rate)

            model_aem = tf.keras.Model(inputs=aem, outputs=body_aem, name="model_aem")
            model_input.append(model_aem.input)
            model_output.append(model_aem.output)

        if input_gravity is not None:
            gravity = tf.keras.Input(shape=input_gravity, name="Gravity")

            if data_augmentation:
                gravity = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

            if is_a_cnn:
                body_gravity = convolutional_body_of_the_cnn(
                    input_layer=gravity,
                    neuron_list=neuron_list,
                    kernel_size=kernel_gravity,
                    dropout=dropout_rate,
                    kernel_regularizes=regularization,
                    pool_size=pool_size,
                )
            else:
                body_gravity = dense_nodes(input_layer=gravity, neuron_list=neuron_list, dropout=dropout_rate)

            model_gravity = tf.keras.Model(inputs=gravity, outputs=body_gravity, name="model_gravity")
            model_input.append(model_gravity.input)
            model_output.append(model_gravity.output)

        if input_magnetic is not None:
            magnetic = tf.keras.Input(shape=input_magnetic, name="Magnetic")

            if data_augmentation:
                magnetic = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

            if is_a_cnn:
                body_magnetic = convolutional_body_of_the_cnn(
                    input_layer=magnetic,
                    neuron_list=neuron_list,
                    kernel_size=kernel_magnetic,
                    dropout=dropout_rate,
                    kernel_regularizes=regularization,
                    pool_size=pool_size,
                )
            else:
                body_magnetic = dense_nodes(input_layer=magnetic, neuron_list=neuron_list, dropout=dropout_rate)

            model_magnetic = tf.keras.Model(inputs=magnetic, outputs=body_magnetic, name="model_magnetic")
            model_input.append(model_magnetic.input)
            model_output.append(model_magnetic.output)

        if input_radiometric is not None:
            radiometric = tf.keras.Input(shape=input_radiometric, name="Radiometric")

            if data_augmentation:
                radiometric = tf.keras.layers.RandomRotation((-0.2, 0.5))(input_layer)

            if is_a_cnn:
                body_radiometric = convolutional_body_of_the_cnn(
                    input_layer=radiometric,
                    neuron_list=neuron_list,
                    kernel_size=kernel_radiometric,
                    dropout=dropout_rate,
                    kernel_regularizes=regularization,
                    pool_size=pool_size,
                )
            else:
                body_radiometric = dense_nodes(input_layer=radiometric, neuron_list=neuron_list, dropout=dropout_rate)

            model_magnetic = tf.keras.Model(inputs=radiometric, outputs=body_radiometric, name="model_radiopmetric")
            model_input.append(model_magnetic.input)
            model_output.append(model_magnetic.output)

        # combined
        combined = tf.keras.layers.Concatenate(axis=-1)(model_output)

        classifier = tf.keras.layers.Dense(
            output,
            activation=last_activation,
            kernel_regularizer=regularization,
            bias_regularizer=None,
            name="final_classifier",
        )(combined)

        model = tf.keras.Model(inputs=model_input, outputs=classifier, name="eis_multimodal")

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return model


@beartype
def make_prediction(
    compiled_model: tf.keras.Model,
    dictionary_of_training: dict,
    dictionary_of_validation: dict,
    training_labels: numpy.ndarray,
    validation_labels: numpy.ndarray,
    epochs: int,
    batch_size: int,
) -> Union[tf.keras.Model, dict, float, int or float, int]:
    """
    Do predictions of the model.

    Parameters:
    compiled_model: an instance of the model.
    dictionary_of_training: a dictionary with training data,
    dictionary_of_validation: a dictionary with training or validation data,
    training_labels: label of training data,
    validation_labels: label of validation or test data.
    epochs: number of epochs for running the model.
    batch_size: batch size to feed the model.

    Return:
        return the compiled model, the score, predictions validation

    Raises:
        TODO
    """

    history = compiled_model.fit(
        dictionary_of_training,
        training_labels,
        validation_data=(dictionary_of_validation, validation_labels),
        batch_size=batch_size,
        epochs=epochs,
    )

    score = compiled_model.evaluate(dictionary_of_validation, validation_labels)
    prediction = compiled_model.predict(dictionary_of_validation)

    return compiled_model, history, score[0], prediction, validation_labels[0]
