from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from keras import backend as K

# from keras.losses import mean_squared_error
# from keras.metrics import binary_crossentropy
from keras.models import Model
from skimage.metrics import structural_similarity as ssim

from eis_toolkit.prediction.machine_learning_general import predict


@beartype
def reshape(data: np.ndarray, shape: tuple | int) -> np.ndarray | None:
    """
    Reshapes the provided data to the specified shape.

    Parameters:
        data: Input data to be reshaped.
        shape: Desired shape.

    Returns:
        Reshaped data.
    """
    try:
        data = data.reshape(shape)
        return data
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return None


@beartype
def preview(model: Model, data: np.ndarray, max_display: int = 10) -> None:
    """
    Reshapes the provided data to the specified shape.

    Parameters:
        model: Model to predict and preview on.
        data: Data to predict and preview.
        max_display: The maximum number of samples to show.

    Returns:
        A matplotlib table showcasing input vs predictions.
    """
    if len(data) > max_display:
        data = data[:max_display]

    # Get predictions
    predictions = predict(data, model)

    # Create a subplot of 2 rows (for original and reconstruction) and columns equal to number of samples
    n_samples = len(data)
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))

    for i in range(n_samples):
        # Display original
        ax = axes[0, i]
        ax.imshow(data[i].squeeze(), cmap="gray" if data[i].shape[-1] == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Original")

        # Display reconstruction
        ax = axes[1, i]
        ax.imshow(predictions[i].squeeze(), cmap="gray" if predictions[i].shape[-1] == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Reconstructed")

    plt.show()


@beartype
def evaluate(model: Model, test_data: np.ndarray) -> Tuple[float, float]:
    """
    Compute the MSE (Mean squared error) and SSIM (Structural similarity index) for the set of test images.

    Parameters:
        model: Trained autoencoder model.
        test_data: List or numpy array of test images.

    Returns:
        Average MSE and average SSIM for the test set.
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
    dataset: (Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray]), input_shape: Tuple[int, ...], is_unet: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray]:
    """
    Prepare the dataset based on the model's expected input shape.

    Parameters:
        dataset: Input data in the format (x_train, x_test).
        input_shape: Expected input shape of the model excluding the batch size.
        is_unet: Whether the target model is a U-Net style model or not.

    Returns:
        Reshaped training and test data.
    """
    if len(dataset) == 2:
        x_train, x_test = dataset
        x_test = x_test.astype("float32") / 255.0
    else:
        x_train = dataset[0]
        x_test = None

    x_train = x_train.astype("float32") / 255.0

    # If it's a U-Net model or a multi-modal regular autoencoder, split the channels
    if is_unet or (len(input_shape) == 3 and input_shape[2] > 1):
        x_train = [x_train[..., i : i + 1] for i in range(input_shape[2])]  # noqa: E203
        if x_test is not None:
            x_test = [x_test[..., i : i + 1] for i in range(input_shape[2])]  # noqa: E203

    # Handle case with no test data
    if x_test is None:
        return (x_train,)

    return x_train, x_test


"""
Usage

REGULAR AUTOENCODER
> x_train_regular, x_test_regular = prepare_data_for_model((x_train_data, x_test_data), input_shape_regular)
> model = build_autoencoder(input_shape, modality)
> model.compile(optimizer='adam', loss='binary_crossentropy') # or loss='mse', if data range is not between 0 and 1
    (in that case, consider also changing the autoencoder model's output activation of 'sigmoid' to 'tanh' or similar
> model = train(model, x_train_regular)

U-NET AUTOECNDOER
> x_train_unet, x_test_unet = prepare_data_for_model((x_train_data, x_test_data), input_shape_unet, is_unet=True)
> model = build_autoencoder_u_net(resolution, modality)
> model.compile(optimizer='adam', loss='binary_crossentropy') # or loss='mse', if data range is not between 0 and 1
    (in that case, consider also changing the model's output activation of 'sigmoid' to 'tanh' or similar
> model = train(model, x_train_unet)
"""
