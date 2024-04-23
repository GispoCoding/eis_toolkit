import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype

from eis_toolkit.exceptions import InvalidDatasetException, InvalidDataShapeException


@beartype
def plot_nn_model_accuracy(model_history: dict) -> plt.Axes:
    """Plot training and validation accuracies for a neural network model.

    Args:
        model_history: Dictionary containing neural network model training history information,
            specifically entries for "accuracy" and "val_accuracy".

    Returns:
        Matplotlib axes containing the produced plot.

    Raises:
        InvalidDatasetException: Raised if "accuracy" or "val_accuracy" are not found in the model_history.
        InvalidDataShapeException: Raised if "accuracy" and "val_accuracy" have mismatching lengths.
    """
    if not all(key in model_history for key in ("accuracy", "val_accuracy")):
        raise InvalidDatasetException("Expected 'accuracy' and 'val_accuracy' to be found in model_history.")
    if len(model_history["accuracy"]) != len(model_history["val_accuracy"]):
        raise InvalidDataShapeException("Expected 'accuracy' and 'val_accuracy' to have the same length.")

    df = pd.DataFrame(
        {
            "Training set accuracy": model_history["accuracy"],
            "Validation set accuracy": model_history["val_accuracy"],
        }
    )
    ax = sns.lineplot(data=df)
    ax.set(xlabel="Epoch", ylabel="Accuracy")

    return ax


@beartype
def plot_nn_model_loss(model_history: dict) -> plt.Axes:
    """Plot training and validation losses for a neural network model.

    Args:
        model_history: Dictionary containing neural network model training history information,
            specifically entries for "loss" and "val_loss".

    Returns:
        Matplotlib axes containing the produced plot.

    Raises:
        InvalidDatasetException: Raised if "loss" or "val_loss" are not found in the model_history.
        InvalidDataShapeException: Raised if "loss" and "val_loss" have mismatching lengths.
    """
    if not all(key in model_history for key in ("loss", "val_loss")):
        raise InvalidDatasetException("Expected 'loss' and 'val_loss' to be found in model_history.")
    if len(model_history["loss"]) != len(model_history["val_loss"]):
        raise InvalidDataShapeException("Expected 'loss' and 'val_loss' to have the same length.")

    df = pd.DataFrame(
        {
            "Training set loss": model_history["loss"],
            "Validation set loss": model_history["val_loss"],
        }
    )
    ax = sns.lineplot(data=df)
    ax.set(xlabel="Epoch", ylabel="Loss")
    return ax
