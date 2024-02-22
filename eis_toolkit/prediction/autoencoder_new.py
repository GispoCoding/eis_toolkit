from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from beartype import beartype
from keras.callbacks import EarlyStopping
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    Multiply,
    UpSampling2D,
    concatenate,
)
from keras.models import Model
from keras.regularizers import l1

# NOTES:
# - Change modality -> band ?
# - With current parameters, these models are classifiers? Should name accordingly and/or create regression models?


@beartype
def _scale_tensors(input_images: Sequence[tf.Tensor], multipliers: Sequence[float]) -> Sequence[tf.Tensor]:
    """
    Scales each tensor in the input based on the given multipliers.

    Parameters
        input_images: List of input tensors
        multipliers: List of scaling factors

    Returns:
        Scaled tensors.
    """
    multipliers_const = tf.constant(multipliers, dtype=tf.float32)

    return [input_images[i] * multipliers_const[i] for i in range(len(input_images))]


@beartype
def _attention_block_skip(x: tf.Tensor, g: tf.Tensor, inter_channel: int) -> tf.Tensor:
    """Implement an attention block with a skip connection.

    Parameters:
        x: The input feature map.
        g: The gating signal.
        inter_channel: Number of filters for the intermediate convolutional layers.

    Returns:
        Output feature map after the attention block.
    """
    # Linear transformation of the input to create new feature map of the input with inter_channel filters
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    # Linear transformation of the gating signal (phi operation) and creates a feature map with inter_channel filters
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    # Add the transformed input feature map and the transformed gating signal
    # Apply the ReLU activation function and then combine the input and the gating signal
    f = Activation("relu")(Add()([theta_x, phi_g]))

    # Reduce the channel dimension of the fused feature map to 1 using a 1x1 convolution
    # Generates the attention coefficients
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    # Apply the sigmoid activation function to the attention coefficients
    # Results in values between 0 and 1, showcasing the attention scores
    rate = Activation("sigmoid")(psi_f)

    # Multiply the original input feature map by the attention scores
    # Which amplifies the features in the input where the att* scores are high
    att_x = Multiply()([x, rate])

    # Return the modified feature map after applying attention
    return att_x


@beartype
def train_autoencoder_regular(
    X: np.ndarray | List[np.ndarray],  # NOTE CHECK LIST
    y: Optional[np.ndarray | List[np.ndarray]],  # NOTE CHECK LIST
    input_shape: tuple,
    modality: int = 1,
    dropout: float = 0.2,
    regularization: float = 0,
    number_of_layers: int = 2,
    filter_size_start: int = 16,
    epochs: int = 50,
    batch_size: int = 128,
    validation_split: float = 0.1,
    validation_data: None | np.ndarray | Tuple[np.ndarray, np.ndarray] = None,
    shuffle: bool = True,
    early_stopping: bool = True,
) -> Tuple[Model, dict]:
    """
    Build an autoencoder model that can handle multiple modalities/bands.

    Parameters:
        X: Training data.
        y: Target labels. In case of autoencoders, it's optional as the targets are often the input.

        (! autoencoder build)
        input_shape: Shape of the input data (excluding batch dimension).
        number_of_layers: Number of layers in encoder and decoder.
        filter_size_start: Initial number of filters in the encoder.
        dropout: Dropout rate to apply between layers.
        regularization: Regularization strength for L1 regularization.
        modality: Number of modalities or bands.

        (!train)
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        validation_split: Fraction of the training data to be used as validation data. Only used when no explicit
            validation_data is provided.
        validation_data: Validation data, either as a tuple of (x_val, y_val) or x_val.
        shuffle: Whether to shuffle the samples at each epoch.
        early_stopping: Whether to use early stopping.

    Returns:
        Trained autoencoder model and training history..
    """
    # 1. Check input data TODO

    # 2. Build and compile autoencoder model
    input_images = [Input(shape=input_shape) for _ in range(modality)]
    # Encoding
    encoded_layers = []
    for image in input_images:
        x = image
        current_filter_size = filter_size_start
        for _ in range(number_of_layers):
            x = Conv2D(
                current_filter_size, (3, 3), activation="relu", padding="same", kernel_regularizer=l1(regularization)
            )(x)
            x = Dropout(dropout)(x)
            x = MaxPooling2D((2, 2), padding="same")(x)
            current_filter_size *= 2
        encoded_layers.append(x)

    x = concatenate(encoded_layers, axis=-1) if modality > 1 else encoded_layers[0]

    # Decoding
    current_filter_size //= 2
    for _ in range(number_of_layers):
        x = Conv2D(
            current_filter_size, (3, 3), activation="relu", padding="same", kernel_regularizer=l1(regularization)
        )(x)
        x = Dropout(dropout)(x)
        x = UpSampling2D((2, 2))(x)
        current_filter_size //= 2

    decoded = Conv2D(
        input_shape[-1], (3, 3), activation="sigmoid", padding="same", kernel_regularizer=l1(regularization)
    )(x)

    model = Model(input_images, decoded)

    model.compile(optimizer="adam", loss="binary_crossentropy")

    # 3. Train the model
    # If no y, targets == inputs
    if y is None:
        y = X

    # Check if using U-Net (multi-modal) or regular autoencoder
    if isinstance(X, list) and len(y) > 1:
        # U-Net multi-modal training
        y = y[0]

    # If validation data is provided and there are not targets, copy input like trainig data
    if validation_data and len(validation_data) == 1:
        validation_data = (validation_data, validation_data)

    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)] if early_stopping else []

    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_data=validation_data if validation_data else None,
        validation_split=validation_split if not validation_data else None,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=callbacks,
    )

    return model, history.history


@beartype
def train_autoencoder_unet(
    X: np.ndarray | List[np.ndarray],  # NOTE CHECK LIST
    y: Optional[np.ndarray | List[np.ndarray]],  # NOTE CHECK LIST
    resolution: int,
    modality: int,
    dropout: float = 0.2,
    regularization: float = 0,
    modality_multipliers: Optional[List[int]] = None,
    number_of_layers: int = 2,
    filter_size_start: int = 8,
    epochs: int = 50,
    batch_size: int = 128,
    validation_split: float = 0.1,
    validation_data: None | np.ndarray | Tuple[np.ndarray, np.ndarray] = None,
    shuffle: bool = True,
    early_stopping: bool = True,
) -> Tuple[Model, dict]:
    """
    Build and trains a U-Net architecture with attention blocks and support for multiple modalities/bands.

    Recommended when regular autoencoder cannot perform well with a task, or when having multiple modalities,

    Parameters:
        X: Training data.
        y: Target labels. In case of autoencoders, it's optional as the targets are often the input.

        (!unet build)
        resolution: Image resolution for each modality.
        modality: The number of modalities or bands.
        dropout: Dropout rate.
        regularization: Regularization rate for L1.
        modality_multipliers: Multipliers for each modality.
        number_of_layers: Number of layers in the encoder/decoder.
        filter_size_start: Initial filter size.

        (!train)
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        validation_split: Fraction of the training data to be used as validation data. Only used when no explicit
            validation_data is provided.
        validation_data: validation data, either as a tuple of (x_val, y_val) or x_val.
        shuffle: Whether to shuffle the samples at each epoch.
        early_stopping: Whether to use early stopping.

    Returns:
        Trained autoencoder model and training history.
    """
    # 1. Check input data TODO

    # 2. Build and compile autoencoder model
    input_images: List[Any] = [Input(shape=(resolution, resolution, 1)) for _ in range(modality)]

    # Scaling
    modality_multipliers = [1 for _ in range(modality)] if modality_multipliers is None else modality_multipliers
    scaled_images = _scale_tensors(input_images, modality_multipliers)

    # Encoder
    skip_connections = []
    encoded_images = []
    for input_img in scaled_images:
        x = input_img
        current_filter_size = filter_size_start
        for i in range(number_of_layers):
            x = Conv2D(current_filter_size, (3, 3), padding="same", kernel_regularizer=l1(regularization))(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            # Store the encoder output for skip connections
            skip_connections.append(x)

            x = MaxPooling2D((2, 2), padding="same")(x)

            # Double the filter size for the next layer
            current_filter_size *= 2

        encoded_images.append(x)

    x = concatenate(encoded_images, axis=-1)

    # Decoder
    current_filter_size = filter_size_start * number_of_layers
    for i in range(number_of_layers):
        x = Conv2D(current_filter_size, (3, 3), padding="same", kernel_regularizer=l1(regularization))(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)

        # Get the corresponding encoder output for skip connection
        skip = skip_connections[-(i + 1)]

        # Apply the attention block to skip connection
        x = _attention_block_skip(x, skip, current_filter_size)

        # Halve the filter size for the next layer
        current_filter_size //= 2

    # Output layer
    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input_images, decoded)

    model.compile(optimizer="adam", loss="binary_crossentropy")

    # 3. Train the model
    # If no y, targets == inputs
    if y is None:
        y = X

    # Check if using U-Net (multi-modal) or regular autoencoder
    if isinstance(X, list) and len(y) > 1:
        # U-Net multi-modal training
        y = y[0]

    # If validation data is provided and there are not targets, copy input like trainig data
    if validation_data and len(validation_data) == 1:
        validation_data = (validation_data, validation_data)

    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)] if early_stopping else []

    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_data=validation_data if validation_data else None,
        validation_split=validation_split if not validation_data else None,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=callbacks,
    )

    return model, history.history
