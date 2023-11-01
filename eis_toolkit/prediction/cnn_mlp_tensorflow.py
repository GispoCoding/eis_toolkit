import os
import random
from typing import Literal, Union, Any

import joblib
import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from beartype import beartype
from osgeo import gdal
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import (NoSuchPathOrDirectory, WrongWindowSize, InvalidDatasetException, CNNException,
                                    InvalidArgumentTypeException, CNNRunningParameterException)
from eis_toolkit.prediction.model_performance_estimation import performance_model_estimation


@beartype
def parse_the_master_file(master_file_path) -> dict:
    """
    Load sets of windows from its folder.

    Parameters:
        master_file_path: path of the masterfile.

    Return:
        Dictionary that contains the following information:
                current_path: this the path of the feature.
                no_data: no data value.
                no_data_value: which value to substitute to the no data.
                min_range_value: min range of the raster.
                max_range_value: max range of the.
                channel: band.
                windows_dimension: the window dimension.
                valid_bands: valid band to use.
                loaded_dataset_as_array: current raster array.
                current_geo_transform: current set of N and E.
                model_type: this is not used here.
    Raises:
        NoSuchPathOrDirectory when some path point to a not such file or directory.
    """

    if not os.path.isfile(master_file_path):
        raise NoSuchPathOrDirectory

    current_dataset = dict()
    # open the handler
    handler = open(master_file_path)

    # loop inside each rows
    for line in handler.readlines():
        line_to_split = line.strip()

        # get band list and convert to int
        bands = line_to_split.split(":")[1].split(",")

        # parse features to int
        bands = [int(x) for x in bands]

        # get other values
        others_values = line_to_split.split(":")[0].split(",")

        # che sub raster from raster
        loaded_raster = gdal.Open(f"{others_values[0]}", gdal.GA_ReadOnly)
        geo = loaded_raster.GetGeoTransform()

        # create a key holder for the dict of feat
        if others_values[1] not in current_dataset.keys():
            current_dataset[others_values[1]] = list()

            current_dataset[others_values[1]].append(
                {
                    "full_path":f"{others_values[0]}",
                    "current_path": f"{others_values[0].split('/')[-2]}/{others_values[0].split('/')[-1]}",
                    "no_data": float(others_values[3]) if others_values[3] != "" else "",
                    "no_data_value": float(others_values[4]) if others_values[4] != "" else 255,
                    "min_range_value": float(others_values[5]) if others_values[5] != "" else "",
                    "max_range_value": float(others_values[6]) if others_values[6] != "" else "",
                    "channel": others_values[1],
                    "windows_dimension": int(others_values[2]),
                    "valid_bands": bands,
                    "loaded_dataset_as_array": loaded_raster.ReadAsArray()[bands, :, :]
                    if loaded_raster.ReadAsArray().ndim > 2
                    else loaded_raster.ReadAsArray(),
                    "current_geo_transform": geo,
                    "model_type": others_values[7],
                }
            )
    handler.close()
    return current_dataset


@beartype
def return_list_of_N_and_E(path_to_data: str) -> list[list[Any]]:
    """
    Load the list of N and E coordinates.

    Parameters:
        input_path: this is what in keras is called optimizer.

    Return:
        a list that contain all N end E

    Raises:
        NoSuchPathOrDirectory when some path point to a not such file or directory.
    """

    if not os.path.isfile(path_to_data):
        raise NoSuchPathOrDirectory

    # load the csv file with the deposit annotation
    handler = open(path_to_data, "r")
    coords = list()
    for row_counter, row in enumerate(handler.readlines()):
        if row_counter != 0:
            coords.append([row.strip().split(",")[-2], row.strip().split(",")[-3]])
    return coords


@beartype
def create_windows_based_of_geo_coords(
    current_raster_object: dict,
    current_E: float,
    current_N: float,
    desired_windows_dimension: int,
    current_loaded_raster: gdal.Dataset or None,
) -> np.ndarray:
    """
    Create windows from geo coordinates.

    Parameters:
       current_raster_object: raster iformation from the masterfile.
       current_E: float point showing the E.
       current_N: float point showing the N.
       desired_windows_dimension: int dimension of the window.
       current_loaded_raster: load tif with gdal

    Return:
        numpy array with the windows inside.

    """

    if current_loaded_raster is None:
        # get the loaded raster and the pix
        current_raster = gdal.Open(current_raster_object["current_path"])
    else:
        current_raster = current_loaded_raster

    spatial_pixel_resolution = current_raster_object["current_geo_transform"][1]

    # get the coords
    start_N = current_N + (desired_windows_dimension / 2) * spatial_pixel_resolution
    end_N = start_N + desired_windows_dimension * spatial_pixel_resolution

    start_E = current_E - (desired_windows_dimension / 2) * spatial_pixel_resolution
    end_E = start_E + desired_windows_dimension * spatial_pixel_resolution

    raster = gdal.Warp(
        "",
        current_raster,
        outputBounds=[start_E, end_N, end_E, start_N],
        format="MEM",
        xRes=spatial_pixel_resolution,
        yRes=-spatial_pixel_resolution,
    )

    values_with_need = raster.ReadAsArray()

    # create the window I m testing with float
    window = (
        np.array(values_with_need).astype("float32").reshape((desired_windows_dimension, desired_windows_dimension, -1))
    )

    # remove no data value:
    if current_raster_object["no_data"] != "":
        window[window == current_raster_object["no_data"]] = current_raster_object["no_data_value"]

    return window

@beartype
def dataset_loader(
    deposit_path: str, unlabelled_data_path: str, desired_windows_dimension: int, path_of_features: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Do the load of the data.

    Parameters:
        deposit_path: the path to the deposit csv file.
        unlabelled_data_path: the path from the 2M csv file.
        desired_windows_dimension: the desired windows dimension.
        path_of_features: masterfile path.

    Return:
        dataset holder and labels holder.

    Raises:
        NoSuchPathOrDirectory: when some path point to a not such file or directory.
        WrongWindowSize:When the size of the windows is <= 1.
    """

    if not os.path.isfile(deposit_path) or not os.path.isfile(unlabelled_data_path):
        raise NoSuchPathOrDirectory

    if desired_windows_dimension <= 1:
        raise WrongWindowSize

    dataset_holder = {}
    labels_holder = list()
    already_done = list()
    same_point = True

    # load the csv file with the deposit annotation
    coordinates_of_deposit = return_list_of_N_and_E(path_to_data=f"{deposit_path}")
    coordinates_of_unlabelled_data = return_list_of_N_and_E(path_to_data=f"{unlabelled_data_path}")

    # parse the master
    current_dataset = parse_the_master_file(master_file_path=path_of_features)

    for key, val in current_dataset.items():
        # create a list that hold the windows
        temp_holder = list()
        concatenated = None
        for tif_obj in current_dataset[key]:
            current_raster = gdal.Open(tif_obj['full_path'])
            for windows_counter, (N, E) in enumerate(coordinates_of_deposit):
                windows = create_windows_based_of_geo_coords(
                    current_raster_object=tif_obj,
                    current_E=float(E),
                    current_N=float(N),
                    desired_windows_dimension=desired_windows_dimension,
                    current_loaded_raster=current_raster,
                )
                temp_holder.append(windows)
                labels_holder.append(1)

            # generate 17 random windows
            rn = 0
            for i in range(0, len(coordinates_of_deposit)):
                if not same_point:
                    # loop until you find a free number
                    while rn in already_done:
                        rn = random.randint(0, len(coordinates_of_unlabelled_data))
                    # add here rn so we do not pick the same windows two time
                    already_done.append(rn)

                # get the actual windows
                windows = create_windows_based_of_geo_coords(
                    current_raster_object=tif_obj,
                    current_E=float(coordinates_of_unlabelled_data[rn][0]),
                    current_N=float(coordinates_of_unlabelled_data[rn][1]),
                    desired_windows_dimension=desired_windows_dimension,
                    current_loaded_raster=current_raster,
                )

                temp_holder.append(windows)
                labels_holder.append(0)
            # concatenate the data
            if concatenated is None:
                concatenated = np.array(temp_holder).astype("float32")

            else:
                concatenated = np.concatenate((concatenated, np.array(temp_holder)), axis=-1)
        dataset_holder[f"{key}"] = concatenated.astype("float32")
    labels_holder = np.array(labels_holder).astype("int")
    return dataset_holder, labels_holder


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
        InvalidDatasetException: when the dataset is null.
    """

    if len(data_dictionary.keys()) <= 0:
        raise InvalidDatasetException

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
        InvalidDatasetException: when the dataset is null.
    """

    if data_to_normalize.shape[0] <= 0:
        raise InvalidDatasetException

    number_of_samples, h, w, c = data_to_normalize.shape
    temp = data_to_normalize.reshape(-1, data_to_normalize.shape[-1])
    normalized_input = normalizator.transform(temp)
    return normalized_input.reshape(number_of_samples, h, w, c)




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
        CNNException: this exception is raised if the input is null.
        InvalidArgumentTypeException: when a parameters i wrong or <= 0.
    """

    if input_layer is None:
        raise CNNException

    if len(neuron_list) <= 0 or kernel_size[0] == 0 or pool_size <= 0 or dropout <= 0:
        raise InvalidArgumentTypeException

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
        CNNException: this exception is raised if the input is null.
        InvalidArgumentTypeException: when a parameters i wrong or <= 0.
    """

    if input_layer is None:
        raise CNNException

    if len(neuron_list) <= 0 or dropout <= 0:
        raise InvalidArgumentTypeException

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
    dropout_rate: Union[None, float] = None,
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
        InvalidArgumentTypeException: one of the arguments is invalid.
    """

    if len(neuron_list) <= 0 or inputs <= 0 or pool_size <= 0:
        raise InvalidArgumentTypeException

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


def make_prediction(
    compiled_model: tf.keras.Model,
    dictionary_of_training: dict,
    dictionary_of_validation: dict,
    training_labels: numpy.ndarray,
    validation_labels: numpy.ndarray,
    epochs: int,
    batch_size: int,
    sample_weights: bool = True,
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
    sample_weights: if you want to sample the weights
    Return:
        return the compiled model, the score, predictions validation

    Raises:
        CNNException: if the compiled model is null.
        InvalidDatasetException: if the dataset is null.
        CNNRunningParameterException: parameters like epochs and batch ize can be <= 0
    """

    if compiled_model is None:
        raise CNNException

    if epochs <= 0:
        raise CNNRunningParameterException

    if len(dictionary_of_training.keys()) <= 0 or len(dictionary_of_validation.keys()) <= 0:
        raise InvalidDatasetException

    if training_labels.shape[0] <= 0 or validation_labels.shape[0] <= 0:
        raise InvalidDatasetException

    history = compiled_model.fit(
        dictionary_of_training,
        training_labels,
        validation_data=(dictionary_of_validation, validation_labels),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight=compute_sample_weight("balanced", training_labels) if sample_weights is not False else None,
    )

    score = compiled_model.evaluate(dictionary_of_validation, validation_labels)
    prediction = compiled_model.predict(dictionary_of_validation)

    return compiled_model, history, score[0], prediction, validation_labels[0]



def do_training_and_prediction_of_the_model(
    deposit_path: str,
    unlabelled_data_path: str,
    path_to_features: str,
    desired_windows_dimension: int = 5,
    cnn_configuration: dict = None,
    threshold: float = 0,
    dump: bool = False,
    epoches: int = 32,
) -> tuple[pd.DataFrame, tf.keras.Model]:
    """
    Do training and evaluation of the model with cross validation.

    Parameters:
        deposit_path: the poath to the csv with 17 points.
        unlabelled_data_path: path to the csv file with 2M points.
        path_to_features: this is a path to a masterfile that contain how to manipulate raster.
        desired_windows_dimension: dimension of the windows.,
        cnn_configuration: all parameters needed for the CNN oe MLP,
        threshold: if you use sigmoid this should be > than 0
        dump: if you want to save the confusion matrix,
        epoches: number of epochs
    Return:
        return pd dataframe that contains the confusion matrix and instance of the best model.

    Raises:
        NoSuchPathOrDirectory when some path point to a not such file or directory.
    """

    if not os.path.isfile(deposit_path) or not os.path.isfile(unlabelled_data_path):
        raise NoSuchPathOrDirectory


    stacked_true, stacked_prediction = list(), list()
    best_score = 0
    model_to_return = None

    # initial cnn config
    if cnn_configuration is None:
        cnn_configuration = {
            "input_aem": None,
            "kernel_aem": None,
            "input_gravity": None,
            "kernel_gravity": None,
            "input_magnetic": None,
            "kernel_magnetic": None,
            "input_radiometric": None,
            "kernel_radiometric": None,
            "regularization": tf.keras.regularizers.L2(0.06),
            "data_augmentation": None,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "inputs": 4,
            "neuron_list": [8, 16],
            "pool_size": 1,
            "dropout_rate": 0.6,
            "output": 2,
            "is_a_cnn": True,
            "last_activation": "softmax",
        }

    windows_holder, labels_holder = dataset_loader(
        deposit_path=deposit_path,
        unlabelled_data_path=unlabelled_data_path,
        desired_windows_dimension=desired_windows_dimension,
        path_of_features=path_to_features,
    )

    # prepare the scaler
    scaler_dictionary = create_the_scaler(data_dictionary=windows_holder, dump=False)

    # get cross validation methods
    selected_cs = performance_model_estimation(cross_validation_type="LOOCV", number_of_split=1)

    for i, (train_index, test_index) in enumerate(selected_cs.split(windows_holder)):
        dictionary_of_training = {}
        dictionary_of_validation = {}
        for key in windows_holder.keys():

            cnn_configuration[f"input_{key.lower()}"] = (
                windows_holder[key][train_index].shape[1],
                windows_holder[key][train_index].shape[2],
                windows_holder[key][train_index].shape[3],
            )

            cnn_configuration[f"kernel_{key.lower()}"] = (
                windows_holder[key][train_index].shape[3],
                windows_holder[key][train_index].shape[3],
            )

            dictionary_of_training[f"{key}"] = normalize_the_data(
                data_to_normalize=windows_holder[f"{key}"][train_index], normalizator=scaler_dictionary[key]
            )

            dictionary_of_validation[f"{key}"] = normalize_the_data(
                data_to_normalize=windows_holder[f"{key}"][test_index], normalizator=scaler_dictionary[key]
            )

        # create multimodal cnn
        cnn = create_multi_modal_cnn(**cnn_configuration)
        model, _, score, prediction, true_label = make_prediction(
            compiled_model=cnn,
            dictionary_of_training=dictionary_of_training,
            dictionary_of_validation=dictionary_of_validation,
            training_labels=labels_holder[train_index],
            validation_labels=labels_holder[test_index],
            epochs=epoches,
            batch_size=int(len(windows_holder) / epoches),
            sample_weights=True,
        )

        score = model.evaluate(dictionary_of_validation, labels_holder[test_index])[0]

        if score > best_score:
            best_score = score
            model_to_return = model

        stacked_true.append(true_label)

        if cnn_configuration["last_activation"] != "sofmax":
            stacked_prediction.append(np.argmax(prediction))
        else:
            if prediction[0] <= threshold:
                stacked_prediction.append(0)
            else:
                stacked_prediction.append(1)

    # create a cm
    cm = confusion_matrix(np.array(stacked_true), np.array(stacked_prediction), normalize="all")
    df = pd.DataFrame(cm, columns=["Non deposit", "deposit"], index=["Non deposit", "deposit"])
    # save the ds
    if dump:
        if not os.path.exists("cm"):
            os.makedirs("cm")
        df.to_csv("cm/cm.csv")

    return df, model_to_return


