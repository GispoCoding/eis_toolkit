import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Lambda, Flatten, \
    Reshape, Conv2DTranspose, Activation, concatenate, Multiply, Add, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
#from sklearn.manifold import TSNE
import keras
from keras.metrics import binary_crossentropy
from keras import backend as K
from skimage.metrics import structural_similarity as ssim
from keras.regularizers import l1, l2
from keras.losses import mean_squared_error
#from sklearn.model_selection import train_test_split
from beartype import beartype
from typing import List, Any, Optional, Tuple

"""Model-inference and building functions"""

@beartype
def train(model: Model,
          x_train: np.ndarray | List[np.ndarray],
          y_train: np.ndarray | List[np.ndarray] | None = None,
          epochs: int = 50,
          batch_size: int = 128,
          validation_split: float = 0.1,
          shuffle: bool = True,
          use_early_stopping: bool = True,
          validation_data: None | np.ndarray | Tuple[np.ndarray, np.ndarray] = None) -> Model:
    """
    Train the provided model using the given training data

    Parameters:
    - model (tf.keras.Model): The model to train
    - x_train (numpy.ndarray or list of numpy.ndarray): Training data
    - y_train: target labels, but in case of autoencoders, it's optional as the targets are often the input
    - epochs (int): Number of epochs for training
    - batch_size (int): Batch size for training
    - validation_split (float): Fraction of the training data to be used as validation data. Only used when no explicit
            validation_data is provided
    - shuffle (bool): Whether to shuffle the samples at each epoch
    - use_early_stopping (bool): Whether to use early stopping
    - validation_data: validation data, either as a tuple of (x_val, y_val) or x_val

    Returns:
    - Trained model
    """

    # If no y_train, targets == inputs
    if y_train is None:
        y_train = x_train

    callbacks_list = []

    # EarlyStopping function
    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Value to be monitored
            patience=5,  # Number of epochs with no improvement after which the training will be stopped
            mode='auto',  # 'auto', 'min' or 'max'. In 'auto', algorithm will detect the direction
            restore_best_weights=True  # Whether to restore model weights from the epoch with the best value result
        )
        callbacks_list.append(early_stopping)

    # Check if validation data is provided, and copy the targets to input like training data in case there are no targets
    if validation_data is not None:
        if isinstance(validation_data, tuple) and len(validation_data) == 2:
            x_val, y_val = validation_data
        else:
            x_val = validation_data
            y_val = validation_data
    else:
        x_val, y_val = None, None

    # Check if using U-Net (multi-modal) or regular autoencoder
    if isinstance(x_train, list) and len(x_train) > 1:
        # U-Net multi-modal training
        model.fit(
            x_train, y_train[0],  # U-Net takes multi-modal inputs but has a single output
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(x_val, y_val) if x_val is not None else None,
            validation_split=None if x_val is not None else validation_split,
            callbacks=callbacks_list
        )
    else:
        # Regular autoencoder training
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(x_val, y_val) if x_val is not None else None,
            validation_split=None if x_val is not None else validation_split,
            callbacks=callbacks_list
        )

    return model

@beartype
def build_autoencoder(input_shape: tuple,
                      modality: int = 1,
                      dropout: float = 0.2,
                      regularization: float = 0,
                      number_of_layers: int = 2,
                      filter_size_start: int = 16) -> Model:
    """
    Builds an autoencoder model that can handle multiple modalities/bands.

    Parameters:
    - input_shape (tuple): Shape of the input data (excluding batch dimension).
    - number_of_layers (int): Number of layers in encoder and decoder.
    - filter_size_start (int): Initial number of filters in the encoder.
    - dropout (float): Dropout rate to apply between layers.
    - regularization (float): Regularization strength for L1 regularization.
    - modality (int): Number of modalities or bands.

    Returns:
    - model: Compiled autoencoder model.
    """

    # List to hold all input layers
    input_imgs = [Input(shape=input_shape) for _ in range(modality)]

    encoded_layers = []
    for idx, input_img in enumerate(input_imgs):
        x = input_img
        current_filter_size = filter_size_start
        for i in range(number_of_layers):
            x = Conv2D(current_filter_size, (3, 3), activation='relu', padding='same',
                       kernel_regularizer=l1(regularization))(x)
            x = Dropout(dropout)(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            current_filter_size *= 2
        encoded_layers.append(x)

    x = concatenate(encoded_layers, axis=-1) if modality > 1 else encoded_layers[0]

    # Decoding
    current_filter_size //= 2
    for i in range(number_of_layers):
        x = Conv2D(current_filter_size, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1(regularization))(x)
        x = Dropout(dropout)(x)
        x = UpSampling2D((2, 2))(x)
        current_filter_size //= 2

    decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same',
                     kernel_regularizer=l1(regularization))(x)

    autoencoder = Model(input_imgs, decoded)
    return autoencoder

@beartype
def build_autoencoder_u_net(resolution: int,
                            modality: int,
                            dropout: float = 0.2,
                            regularization: float = 0,
                            modality_multipliers: Optional[List[int]] = None,
                            number_of_layers: int = 2,
                            filter_size_start: int = 8) -> Model:

    """
    Build a U-Net architecture with attention blocks and support for multiple modalities/bands.
    Recommended when regular autoencoder cannot perform well with a task, or when having multiple modalities

    Parameters:
    - resolution: Image resolution for each modality.
    - modality: The number of modalities or bands.
    - dropout: Dropout rate.
    - regularization: Regularization rate for L1.
    - modality_multipliers: Multipliers for each modality.
    - number_of_layers: Number of layers in the encoder/decoder.
    - filter_size_start: Initial filter size.

    Returns:
    - model: Compiled U-Net autoencoder model.
    """

    # List to hold all input layers
    #input_imgs = [Input(shape=(resolution, resolution, 1)) for _ in range(modality)]
    input_imgs: List[Any] = [Input(shape=(resolution, resolution, 1)) for _ in range(modality)]
    skip_connections = []
    encoded_imgs = []

    if modality_multipliers is None:
        multipliers = [1 for _ in range(modality)]
    else:
        multipliers = modality_multipliers

    # Scaling
    scaled_imgs = scale_tensors(input_imgs, multipliers)

    # Encoder
    for idx, input_img in enumerate(scaled_imgs):
        x = input_img
        current_filter_size = filter_size_start
        for i in range(number_of_layers):
            x = Conv2D(current_filter_size, (3, 3), padding='same', kernel_regularizer=l1(regularization))(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # Store the encoder output for skip connections
            skip_connections.append(x)

            x = MaxPooling2D((2, 2), padding='same')(x)
            current_filter_size *= 2  # Double the filter size for the next layer
        encoded_imgs.append(x)

    x = concatenate(encoded_imgs, axis=-1)

    # Decoder
    current_filter_size = filter_size_start * number_of_layers
    for i in range(number_of_layers):
        x = Conv2D(current_filter_size, (3, 3), padding='same', kernel_regularizer=l1(regularization))(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        # Get the corresponding encoder output for skip connection
        skip = skip_connections[-(i + 1)]

        # Apply the attention block to skip connection
        x = attention_block_skip(x, skip, current_filter_size)

        current_filter_size //= 2  # Halve the filter size for the next layer

    # Output layer
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder_multi_channel = Model(input_imgs, decoded)
    return autoencoder_multi_channel

@beartype
def reshape(data: np.ndarray, shape: tuple | int) -> np.ndarray | None:
    """
    Reshapes the provided data to the specified shape

    Parameters:
    - data (numpy.ndarray): Input data to be reshaped
    - shape: Desired shape

    Returns:
    - Reshaped data
    """

    try:
        data = data.reshape(shape)
        return data
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return None

@beartype
def model_predict(model: Model, input: np.ndarray) -> np.ndarray:
    """
    Predict using an autoencoder

    Parameters:
    - model: Trained autoencoder model
    - input: List or numpy array of images to predict (take care to reshape)

    Returns:
    - List of numpy array of predictions
    """

    # Make a prediction
    predicted_image = model.predict(input)

    return predicted_image

@beartype
def preview(model: Model, data: np.ndarray, max_display: int = 10) -> None:
    """
    Reshapes the provided data to the specified shape

    Parameters:
    - model: model to predict and preview on
    - data: Data to predict and preview
    - max_display: the maximum number of samples to show

    Returns:
    - A matplotlib table showcasing input vs predictions
    """

    if len(data) > max_display:
        data = data[:max_display]

    # Get predictions
    predictions = model_predict(model, data)

    # Create a subplot of 2 rows (for original and reconstruction) and columns equal to number of samples
    n_samples = len(data)
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))

    for i in range(n_samples):
        # Display original
        ax = axes[0, i]
        ax.imshow(data[i].squeeze(), cmap='gray' if data[i].shape[-1] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('Original')

        # Display reconstruction
        ax = axes[1, i]
        ax.imshow(predictions[i].squeeze(), cmap='gray' if predictions[i].shape[-1] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('Reconstructed')

    plt.show()

@beartype
def evaluate(model: Model, test_data: np.ndarray) -> Tuple[float, float]:
    """
    Compute the MSE (Mean squared error) and SSIM (Structural similarity index) for the set of test images using the provided model.

    Parameters:
    - model: Trained autoencoder model
    - test_data: List or numpy array of test images

    Returns:
    - Average MSE and average SSIM for the test set
    """

    # Get the autoencoder's predictions
    reconstructed_images = model.predict(test_data)

    # Initialize accumulators for MSE and SSIM
    mse_accumulator = 0.0
    ssim_accumulator = 0.0

    # Compute MSE and SSIM for each image
    for original, reconstructed in zip(test_data, reconstructed_images):
        # MSE
        mse_accumulator += K.mean(K.square(original - reconstructed))

        # Scale the images to be in the range [0,255] for SSIM computation
        original_for_ssim = (original * 255).astype(np.uint8)
        reconstructed_for_ssim = (reconstructed * 255).astype(np.uint8)

        # SSIM (used on 2D grayscale images; adapt as necessary for color images)
        ssim_value, _ = ssim(original_for_ssim.squeeze(), reconstructed_for_ssim.squeeze(), full=True)
        ssim_accumulator += ssim_value

    # Calculate average MSE and SSIM
    avg_mse = mse_accumulator / len(test_data)
    avg_ssim = ssim_accumulator / len(test_data)

    print(f"Average MSE: {avg_mse}, Average SSIM: {avg_ssim}")
    return avg_mse, avg_ssim

@beartype
def prepare_data_for_model(
    dataset: (Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray]),
    input_shape: Tuple[int, ...],
    is_unet: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray]:
    """
    Prepares the dataset based on the model's expected input shape.

    Parameters:
    - dataset (tuple): Input data in the format (x_train, x_test).
    - input_shape (tuple): Expected input shape of the model excluding the batch size.
    - is_unet (bool): Whether the target model is a U-Net style model or not.

    Returns:
    - tuple: Reshaped training and test data.
    """
    if len(dataset) == 2:
        x_train, x_test = dataset
        x_test = x_test.astype('float32') / 255.
    else:
        x_train = dataset[0]
        x_test = None

    x_train = x_train.astype('float32') / 255.

    # If it's a U-Net model or a multi-modal regular autoencoder, split the channels
    if is_unet or (len(input_shape) == 3 and input_shape[2] > 1):
        x_train = [x_train[..., i:i + 1] for i in range(input_shape[2])]
        if x_test is not None:
            x_test = [x_test[..., i:i + 1] for i in range(input_shape[2])]

    # Handle case with no test data
    if x_test is None:
        return x_train,

    return x_train, x_test


"""
_______________________________________

Utility functions
_______________________________________
"""


def scale_tensors(input_imgs, multipliers):
    """
    Scales each tensor in the input based on the given multipliers.

    Parameters
    - input_imgs (list of tf.Tensor): List of input tensors
    - multipliers (list of float): List of scaling factors

    Returns:
    - list of tf.Tensor: Scaled tensors.
    """
    multipliers_const = tf.constant(multipliers, dtype=tf.float32)

    return [input_imgs[i] * multipliers_const[i] for i in range(len(input_imgs))]

@beartype
def attention_block_skip(x, g, inter_channel):
    """
    Implement an attention block with a skip connection.

    Parameters:
    - x (tf.Tensor): The input feature map
    - g (tf.Tensor): The gating signal
    - inter_channel (int): Number of filters for the intermediate convolutional layers

    Returns:
    - tf.Tensor: Output feature map after the attention block
    """

    # Linear transformation of the input to create new feature map of the input with inter_channel filters
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    # Linear transformation of the gating signal (phi operation) and creates a feature map with inter_channel filters
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    # Add the transformed input feature map and the transformed gating signal
    # Apply the ReLU activation function and then combine the input and the gating signal
    f = Activation('relu')(Add()([theta_x, phi_g]))

    # Reduce the channel dimension of the fused feature map to 1 using a 1x1 convolution
    # Generates the attention coefficients
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    # Apply the sigmoid activation function to the attention coefficients
    # Results in values between 0 and 1, showcasing the attention scores
    rate = Activation('sigmoid')(psi_f)

    # Multiply the original input feature map by the attention scores
    # Which amplifies the features in the input where the att* scores are high
    att_x = Multiply()([x, rate])

    # Return the modified feature map after applying attention
    return att_x


"""Usage"""
# Regular autoencoder
# x_train_regular, x_test_regular = prepare_data_for_model((x_train_data, x_test_data), input_shape_regular)
# model = build_autoencoder(input_shape, modality)
# model.compile(optimizer='adam', loss='binary_crossentropy') # or loss='mse', if data range is not between 0 and 1
# (in that case, consider also changing the autoencoder model's output activation of 'sigmoid' to 'tanh' or similar
# model = train(model, x_train_regular)

# U-Net autoencoder
# x_train_unet, x_test_unet = prepare_data_for_model((x_train_data, x_test_data), input_shape_unet, is_unet=True)
# model = build_autoencoder_u_net(resolution, modality)
# model.compile(optimizer='adam', loss='binary_crossentropy') # or loss='mse', if data range is not between 0 and 1
# (in that case, consider also changing the model's output activation of 'sigmoid' to 'tanh' or similar
# model = train(model, x_train_unet)