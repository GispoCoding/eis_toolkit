import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import InvalidDatasetException, InvalidDataShapeException
from eis_toolkit.prediction.mlp import train_MLP_classifier
from eis_toolkit.transformations.one_hot_encoding import one_hot_encode
from eis_toolkit.validation.plot_nn_model_performance import plot_nn_model_accuracy, plot_nn_model_loss

X_IRIS, Y_IRIS = load_iris(return_X_y=True)
X_DIABETES, Y_DIABETES = load_diabetes(return_X_y=True)
SEED = 42

X = StandardScaler().fit_transform(X_IRIS)
y = one_hot_encode(Y_IRIS).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

model, history = train_MLP_classifier(
    X_train,
    y_train,
    neurons=[16],
    output_neurons=3,
    last_activation="softmax",
    loss_function="categorical_crossentropy",
    metrics=["accuracy"],
    random_state=SEED,
)


def test_plot_nn_model_performance():
    """Tests that plotting neural network model accuracy and loss work as expected."""
    ax = plot_nn_model_accuracy(history)
    assert isinstance(ax, plt.Axes)

    ax = plot_nn_model_loss(history)
    assert isinstance(ax, plt.Axes)


def test_plot_nn_model_performance_invalid_history_dict_entires():
    """Test the missing entries in history dictionary raise the correct exceptions."""
    invalid_history = history.copy()
    with pytest.raises(InvalidDatasetException):
        invalid_history.pop("accuracy")
        plot_nn_model_accuracy(invalid_history)

    with pytest.raises(InvalidDatasetException):
        invalid_history.pop("loss")
        plot_nn_model_loss(invalid_history)


def test_plot_nn_model_performance_mismatching_lenghts():
    """Test the mismatching entry lenghts in history dictionary raise the correct exceptions."""
    invalid_history = history.copy()
    with pytest.raises(InvalidDataShapeException):
        invalid_history["accuracy"] = invalid_history["accuracy"][:-1]
        plot_nn_model_accuracy(invalid_history)

    with pytest.raises(InvalidDataShapeException):
        invalid_history["loss"] = invalid_history["loss"][:-1]
        plot_nn_model_loss(invalid_history)
