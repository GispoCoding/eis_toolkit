import glob
import os
from typing import Union

import numpy as np
import numpy.ma as ma
import rasterio
import tensorflow as tf
from beartype import beartype
from keras.optimizers import SGD

from eis_toolkit.exceptions import InvalidNumberOfConv2DLayer, NumericValueSignException


@beartype
def img_loader(image_dir: str) -> (np.ndarray, list, np.ndarray):
    """

     Do the Fetches all the tiffs in the given directory and creates a numpy ndarray of shape.

     Shape type (image_count, bands, width, height) from them. Returns the array, tiff metadata as list and associated
     nodatamasks in shape (image_count, width, height) Tiffs are assumed to be same size and named {number}.tif
    starting from 0

    Parameter:
        image_dir: the directory containing the images

    Returns:
        the numpy ndarray of the tiffs
        the list containing the meta
        no data mask
    """
    # fetching the filepaths
    paths = []
    metas = []
    for path in glob.glob(os.path.join(image_dir, "*.tif")):
        paths.append(path)

    img_count = len(paths)

    # Getting the size of the images
    with rasterio.open(paths[0]) as src:
        meta = src.meta.copy()
    image_width = meta["width"]
    image_height = meta["height"]

    nodata_masks = np.empty((img_count, image_width, image_height), dtype="bool")
    for i in range(img_count):
        path = os.path.join(image_dir, str(i) + ".tif")
        with rasterio.open(path) as src:
            meta = src.meta.copy()
            metas.append(meta)
            # Note: rasterio returns shape (bands, width, height)
            img_arr = np.empty(
                (
                    meta["count"],
                    image_width,
                    image_height,
                ),
                dtype=meta["dtype"],
            )

            for band_n in range(1, meta["count"] + 1):
                nodata_mask = src.read_masks(band_n)
                nodata_mask = nodata_mask == 0
                if band_n == 1:
                    nodata_mask_full = nodata_mask
                else:
                    nodata_mask_full = np.logical_and(nodata_mask_full, nodata_mask, out=nodata_mask_full)

                # normalization
                band_arr = src.read(band_n)
                # Setting nodatamasks to zero
                band_arr[band_arr == -999999] = 0
                band_arr_masked = ma.masked_array(band_arr, mask=nodata_mask)
                minimum = band_arr_masked.min()
                maximum = band_arr_masked.max()
                band_arr = (band_arr_masked - minimum) / (maximum - minimum)
                img_arr[band_n - 1] = band_arr

        nodata_masks[i] = nodata_mask_full
        # Storing array needs to be created only on the first iteration.
        # All images are assumed to be same size and dtype, so the meta of the first one is used
        if i == 0:
            data_arr = np.empty((img_count, meta["count"], meta["width"], meta["height"]), dtype=meta["dtype"])
        data_arr[i] = img_arr

    data_arr = np.moveaxis(data_arr, 1, -1)
    return data_arr, metas, nodata_masks


@beartype
def label_loader(label_dir: str) -> (np.ndarray, int):
    """
    Do the Fetches all tiffs in the given directory and return a numpy ndarray of shape (image_count, width, height).

    The images are assumed to be same size and contain the labels with numbers showing classes.
    The tiffs should have only one band.

    Parameters:
        label_dir: the directory containing the label files

    Return:
        a numpy ndarray containing labels
        no data value

    """
    # fetching the filepaths
    paths = []
    for path in glob.glob(os.path.join(label_dir, "*.tif")):
        paths.append(path)

    img_count = len(paths)

    # Getting the size of the images
    with rasterio.open(paths[0]) as src:
        meta = src.meta.copy()
    nodata_value = meta["nodata"]
    data_arr = np.empty((img_count, meta["count"], meta["width"], meta["height"]), dtype="float32")

    for i in range(img_count):
        path = os.path.join(label_dir, str(i) + ".tif")
        with rasterio.open(path) as src:
            img_arr = src.read(1)
        # Or maybe nodata as zeros is just fine
        img_arr[img_arr == 255] = 0
        # moving band count(1) to last place
        img_arr = img_arr.astype("float32")
        data_arr[i] = img_arr
    data_arr = np.moveaxis(data_arr, 1, -1)
    return data_arr, nodata_value


@beartype
def build_autoencoder_multichannel_with_skip_connection(
    input_shape: tuple,
    kernel_shape: tuple,
    list_of_convolutional_layers: list,
    dropout: float or None,
    pool_size: int,
    up_sampling_factor: int,
    output_filters: int,
    output_kernel: tuple,
    last_activation: str,
    data_augmentation: bool,
    data_augmentation_params_crop: int,
    data_augmentation_params_rotation: tuple = (-0.3, 0.3),
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
) -> tf.keras.Model:
    """

    Do build the Unet.

    Parameters:
        input_shape The shap of the input used by the Unet:
        kernel_shape The shape of the convolution kernel:
        list_of_convolutional_layers The list of convolutional layers of the Unet. This list will be reversed
           for the decoder.
        dropout: This is the dropout rate assigned.
        pool_size: The size of the max pooling layer.
        up_sampling_factor: the decoder need up sampling factor to enlarge the features layer by layer.
        output_filters: the number of filters of the output.
        output_kernel: the dimension of the output.
        last_activation: last activation sigmoid by default.
        data_augmentation: if you want to include data augmentation right before the input layer.
        data_augmentation_params_crop: if data augmentation is true fill this value.
        data_augmentation_params_rotation: if data augmentation is true fill this value.
        regularization: type of regularization for each layer.

    Return:
        the built Unet.
    """
    # List to hold all input layers
    input_img = tf.keras.Input(shape=input_shape)

    if data_augmentation:
        x = tf.keras.layers.RandomFlip()(input_img)
        x = tf.keras.layers.RandomCrop(data_augmentation_params_crop, data_augmentation_params_crop)(x)
        x = tf.keras.layers.RandomRotation(data_augmentation_params_rotation)(x)

    skip_connections = []

    # build the encoder
    for layer_counter, layer in enumerate(list_of_convolutional_layers):
        if layer_counter == 0 and data_augmentation is False:
            x = tf.keras.layers.Conv2D(
                layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization
            )(input_img)
        else:
            x = tf.keras.layers.Conv2D(
                layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization
            )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization)(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        skip_connections.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), padding="same")(x)

    for layer_counter, layer in enumerate(reversed(list_of_convolutional_layers)):
        # Decoder block 1
        # skip_1 = skip_connections[-1]  # Corresponding output from the encoder
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization)(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            int(layer / 2), kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.UpSampling2D((up_sampling_factor, up_sampling_factor))(x)
        x = tf.keras.layers.concatenate([x, skip_connections[-(layer_counter + 1)]], axis=-1)

        if layer_counter == len(list_of_convolutional_layers):

            # Output Layer
            x = tf.keras.layers.Conv2D(
                layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv2D(
                layer, kernel_size=kernel_shape, padding="same", kernel_regularizer=regularization
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

    # activation with normalized tanh should be considered because it has steeper gradients
    decoded = tf.keras.layers.Conv2D(
        output_filters, kernel_size=output_kernel, activation=last_activation, padding="same"
    )(x)

    # Create the model
    autoencoder_multi_channel = tf.keras.Model(input_img, decoded)

    return autoencoder_multi_channel


def dice_coeff_uncertain(
    y_true: np.ndarray, y_pred: np.ndarray, uncertainmask: np.ndarray, smooth: float = 1e-6
) -> (float, np.ndarray):
    """
    From Wu, S., Heitzler, M. & Hurni, L. (2022).

    Leveraging uncertainty estimation and spatial pyramid pooling for extracting hydrological features from scanned
    historical topographic maps’, GIScience & Remote Sensing, 59(1), pp. 200–214. Available
    at: https://doi.org/10.1080/15481603.2021.2023840.

    Parameters:
        y_true: the true labels of the dataset.
        y_pred: the predicted labels from the model.
        uncertainmask: the uncertainty mask.
        smooth: the smooth factor.

    Return:
        the dice loss value and the mask.
    """

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    if uncertainmask is not None:
        # interpolate the value using the predicted uncertainty
        y_pred = (tf.keras.backend.ones_like(uncertainmask) - uncertainmask) * y_true + uncertainmask * y_pred

    # want the dice coefficient should always be in 0 and 1
    intersection = tf.keras.backend.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)
    mask = tf.keras.backend.cast(
        tf.keras.backend.not_equal(tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection, 0),
        "float32",
    )

    return dice, mask


def regularization_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """

    Do compute the dice loss.

    Parameters:
        y_true: the true labels of the dataset.
        y_pred: the predicted labels from the model.

    Return:
        a floating poit representing the loss.
    """

    y_pred_uncertain = y_pred[:, :, :, 1]
    reg_loss = tf.keras.backend.mean(-tf.keras.backend.log(y_pred_uncertain[:, :, :]))
    return reg_loss


def dice_coefficient(prediction: tf.Tensor, true_label: tf.Tensor) -> tf.Tensor:
    """

    Do calculate the dice loss coefficient.

    Parameters:
        true_label: the true labels of the dataset.
        prediction: the predicted labels from the model.

    Return:
        a floating poit representing the loss.
    """
    prediction = tf.cast(prediction, tf.float32)
    true_label = tf.cast(true_label, tf.float32)
    numerator = 2 * tf.reduce_sum(prediction * true_label)
    divisor = tf.reduce_sum(prediction**2) + tf.reduce_sum(true_label**2)
    return numerator / divisor


def dice_loss(prediction: tf.Tensor, true_label: tf.Tensor) -> tf.Tensor:
    """

    Do calculate the dice loss.

    Parameters:
        true_label: the true labels of the dataset.
        prediction: the predicted labels from the model.

    Return:
        a floating poit representing the loss.

    """

    DC = dice_coefficient(prediction, true_label)
    dice_loss = 1 - DC
    return dice_loss


def dice_loss_uncertain(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
    uncert_coef: float = 0.2,
) -> float or None:
    """
    From Wu, S., Heitzler, M. & Hurni, L. (2022).

    Leveraging uncertainty estimation and spatial pyramid pooling for extracting hydrological features from scanned
    historical topographic maps, GIScience & Remote Sensing, 59(1), pp. 200–214.
    Available at: https://doi.org/10.1080/15481603.2021.2023840.

    Parameters:
        y_true: the predicted labels of the dataset.
        y_pred: the true labels of the dataset.
        num_classes: classes used to calculate the loss.
        smooth: the smoothing coefficient.
        uncert_coef: the uncertainty coefficient.

    Return:
        the loss as a float point number.
    """
    dice = []

    y_true = y_true[:, :, :, 0]
    y_pred_labels = y_pred[:, :, :, 0]
    y_pred_uncertain = y_pred[:, :, :, 1]

    d, m = dice_coeff_uncertain(y_true, y_pred_labels, y_pred_uncertain)
    if m != 0:
        dice.append(d)

    dice_mutilabel = tf.keras.backend.sum(dice) / (len(dice) + smooth)
    uncertain_reg = regularization_loss(y_true, y_pred)
    loss = 1 - dice_mutilabel + uncert_coef * uncertain_reg
    return loss


def train_and_predict_the_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
    is_uncertainty: bool = True,
    list_of_convolutional_layers: list = [32, 64, 128, 256],
    dropout: float = 0.2,
    pool_size: int = 2,
    up_sampling_factor: int = 2,
    output_filters: int = 2,
    output_kernel: tuple = (1, 1),
    last_activation: str = "sigmoid",
    data_augmentation: bool = True,
    data_augmentation_params_crop: int = 28,
    data_augmentation_params_rotation: tuple = (-0.3, 0.3),
    regularization: Union[tf.keras.regularizers.L1, tf.keras.regularizers.L2, tf.keras.regularizers.L1L2, None] = None,
    uncertainty_coefficient: float = 0.2,
) -> np.ndarray:
    """
    Train and predict the Unet.

        Parameters:
            x_train: a numpy array with the training sample inside
            y_train: labels of the training dataset,
            x_test: a numpy array with the testing sample inside,
            y_test: labels of the testing dataset,
            batch_size: how many sample per epochs should be used for fitting the model,
            epochs: how many epochs used for the training phase.
            is_uncertainty: bool = True, if you want to add uncertainty estimation to your Mini-Unet.
            list_of_convolutional_layers: how many convolutional layer suggestion -> [32, 64, 128, 256],
            dropout: This is the dropout rate assigned. It is used to randomly remove some predictions.
            pool_size: The size of the max pooling layer.
            up_sampling_factor: the decoder need up sampling factor to enlarge the features layer by layer.
            output_filters: the number of output nodes.
            output_kernel: the dimension of the output, it should be the same of the input.
            last_activation: last activation sigmoid by default.
            data_augmentation: if you want to include data augmentation right before the input layer.
            data_augmentation_params_crop: if data augmentation is true fill this value (crop range).
            data_augmentation_params_rotation: if data augmentation is true fill this value (rotation range).
            regularization: type of regularization for each layer. (L1, L2, L1L2, or None).
        Raise:
            InvalidInputException: when one input is null.
        Return:
            the predicted numpy array
    """

    if x_train.shape[0] == 0 or x_train is None:
        raise NumericValueSignException

    if x_test.shape[0] == 0 or x_test is None:
        raise NumericValueSignException

    if y_train.shape[0] == 0 or y_train is None:
        raise NumericValueSignException

    if y_test.shape[0] == 0 or y_test is None:
        raise NumericValueSignException

    if len(list_of_convolutional_layers) <= 0:
        raise InvalidNumberOfConv2DLayer

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    model = build_autoencoder_multichannel_with_skip_connection(
        input_shape=x_train.shape[1:],
        kernel_shape=(x_train.shape[-1], x_train.shape[-1]),
        list_of_convolutional_layers=list_of_convolutional_layers,
        dropout=dropout,
        pool_size=pool_size,
        up_sampling_factor=up_sampling_factor,
        output_filters=output_filters,
        output_kernel=output_kernel,
        last_activation=last_activation,
        data_augmentation=data_augmentation,
        data_augmentation_params_crop=data_augmentation_params_crop,
        data_augmentation_params_rotation=data_augmentation_params_rotation,
        regularization=regularization,
    )

    model.compile(
        optimizer=SGD(learning_rate=0.008, momentum=0.9),
        loss=dice_loss_uncertain if is_uncertainty else dice_loss,
    )

    model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=[callback]
    )

    prediction = model.predict(x_test, verbose=1)
    return prediction
