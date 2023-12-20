import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eis_toolkit import exceptions
from eis_toolkit.prediction.mlp import train_MLP_classifier, train_MLP_regressor
from eis_toolkit.transformations.one_hot_encoding import one_hot_encode

X_IRIS, Y_IRIS = load_iris(return_X_y=True)
X_DIABETES, Y_DIABETES = load_diabetes(return_X_y=True)


def test_train_MLP_classifier():
    """Test that training MLP classifier works as expected."""
    X = StandardScaler().fit_transform(X_IRIS)
    y = one_hot_encode(Y_IRIS).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history = train_MLP_classifier(
        X_train,
        y_train,
        neurons=[16],
        output_neurons=3,
        last_activation="softmax",
        loss_function="categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model is not None, "Model should not be None"
    assert "loss" in history, "History should contain loss"
    assert "accuracy" in history, "History should contain accuracy"
    assert len(model.layers) > 0, "Model should have layers"

    scores = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy = scores[1]

    # Can't check if output is equal to a single number as there is some randomness involved in creating MLP model
    assert history["accuracy"][-1] > 0.7, "Last epoch's accuracy should be reasonably high"
    assert test_accuracy > 0.7, "Test accuracy should be reasonably high"


def test_train_MLP_regressor():
    """Test that training MLP regressor works as expected."""
    X = StandardScaler().fit_transform(X_DIABETES)
    y = one_hot_encode(Y_DIABETES).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history = train_MLP_regressor(X_train, y_train, neurons=[16], output_neurons=1, metrics=["mse"])

    assert model is not None, "Model should not be None"
    assert "loss" in history, "History should contain loss"
    assert "mse" in history, "History should contain mse"
    assert len(model.layers) > 0, "Model should have layers"

    scores = model.evaluate(X_test, y_test, verbose=0)
    test_mse = scores[1]

    # Can't check if output is equal to a single number as there is some randomness involved in creating MLP model
    assert history["mse"][-1] < 0.3, "Last epoch's MSE should be reasonably low"
    assert test_mse < 0.3, "Test MSE should be reasonably low"


def test_invalid_neurons():
    """Test that invalid list for neurons raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[])
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[])


def test_invalid_validation_split():
    """Test that invalid value for validation_split raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], validation_split=-1.0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_regressor(X_IRIS, Y_IRIS, neurons=[16], validation_split=-1.0)


def test_invalid_learning_rate():
    """Test that invalid value for learning_rate raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], learning_rate=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], learning_rate=0)


def test_invalid_dropout_rate():
    """Test that invalid value for dropout_rate raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], dropout_rate=2)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], dropout_rate=2)


def test_invalid_es_patience():
    """Test that invalid value for es_patience raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], es_patience=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], es_patience=0)


def test_invalid_batch_size():
    """Test that invalid value for batch_size raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], batch_size=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], batch_size=0)


def test_invalid_epochs():
    """Test that invalid value for epochs raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], epochs=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], epochs=0)


def test_invalid_output_neurons():
    """Test that invalid value for output_neurons raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], output_neurons=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], output_neurons=0)


def test_invalid_output_neurons_for_binary_crossentropy():
    """Test that invalid value for output_neurons for binary crossentropy raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        train_MLP_classifier(X_IRIS, Y_IRIS, neurons=[16], output_neurons=2, loss_function="binary_crossentropy")
