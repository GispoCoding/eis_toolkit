"""Tests for the Bayesian Neural Network module.

This module contains tests for the BayesianNeuralNetworkClassifier class and the
bayesian_neural_network_classifier_train function. It tests both the main functionality
and parameter validation.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import InvalidDataShapeException, InvalidParameterValueException
from eis_toolkit.prediction.bayesian_neural_network import (
    BayesianNeuralNetworkClassifier,
    bayesian_neural_network_classifier_train,
)

X_IRIS, Y_IRIS = load_iris(return_X_y=True)

# Convert to binary classification problem (class 0 vs rest)
Y_IRIS_BINARY = (Y_IRIS > 0).astype(int)
SEED = 42


def test_bayesian_neural_network_classifier():
    """Test that training BayesianNeuralNetworkClassifier works as expected."""
    X = StandardScaler().fit_transform(X_IRIS)
    y = Y_IRIS_BINARY
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = BayesianNeuralNetworkClassifier(
        hidden_units=[16, 8],
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        n_samples=10,
        validation_split=0.1,
        early_stopping_patience=5,
        random_state=SEED,
    )

    model.fit(X_train, y_train)

    assert hasattr(model, "history"), "Model should have history attribute"
    assert "loss" in model.history, "History should contain loss"
    assert len(model.history["loss"]) > 0, "History should have loss values"

    # Test prediction methods
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],), "Predictions should have correct shape"

    y_proba = model.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2), "Probability predictions should have correct shape"

    # Test prediction with uncertainty
    probs, uncertainties = model.predict_with_uncertainty(X_test)
    assert probs.shape == (X_test.shape[0],), "Probabilities should have correct shape"
    assert uncertainties.shape == (X_test.shape[0],), "Uncertainties should have correct shape"


def test_bayesian_neural_network_classifier_train():
    """Test that bayesian_neural_network_classifier_train function works as expected."""
    X = StandardScaler().fit_transform(X_IRIS)
    y = Y_IRIS_BINARY
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = bayesian_neural_network_classifier_train(
        X_train,
        y_train,
        hidden_units=[16, 8],
        learning_rate=0.001,
        epochs=50,
        batch_size=512,
        n_samples=10,
        validation_split=0.1,
        early_stopping_patience=5,
        random_state=SEED,
    )

    assert hasattr(model, "history"), "Model should have history attribute"
    assert "loss" in model.history, "History should contain loss"
    assert len(model.history["loss"]) > 0, "History should have loss values"

    # Test prediction methods
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],), "Predictions should have correct shape"

    y_proba = model.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2), "Probability predictions should have correct shape"

    # Test prediction with uncertainty
    probs, uncertainties = model.predict_with_uncertainty(X_test)
    assert probs.shape == (X_test.shape[0],), "Probabilities should have correct shape"
    assert uncertainties.shape == (X_test.shape[0],), "Uncertainties should have correct shape"


def test_invalid_X_shape():
    """Test that invalid shape for X raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        bayesian_neural_network_classifier_train(X_IRIS.ravel(), Y_IRIS_BINARY)


def test_nonmatching_sample_count():
    """Test that differing number of samples for X and y raises the correct exception."""
    extra_row = np.random.rand(1, X_IRIS.shape[1])
    X_mismatched = np.concatenate((X_IRIS, extra_row), axis=0)
    with pytest.raises(InvalidDataShapeException):
        bayesian_neural_network_classifier_train(X_mismatched, Y_IRIS_BINARY)


def test_invalid_learning_rate():
    """Test that invalid value for learning_rate raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, learning_rate=0)


def test_invalid_epochs():
    """Test that invalid value for epochs raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, epochs=0)


def test_invalid_batch_size():
    """Test that invalid value for batch_size raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, batch_size=0)


def test_invalid_n_samples():
    """Test that invalid value for n_samples raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, n_samples=0)


def test_invalid_validation_split():
    """Test that invalid value for validation_split raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, validation_split=-0.1)
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, validation_split=1.1)


def test_invalid_early_stopping_patience():
    """Test that invalid value for early_stopping_patience raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, early_stopping_patience=0)


def test_invalid_early_stopping_monitor():
    """Test that invalid value for early_stopping_monitor raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, early_stopping_monitor="invalid_monitor")


def test_invalid_clip_norm():
    """Test that invalid value for clip_norm raises the correct exception."""
    # Negative clip_norm should raise an exception
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, clip_norm=-1.0)

    # None is valid (disables clipping)
    model = bayesian_neural_network_classifier_train(
        X_IRIS, 
        Y_IRIS_BINARY, 
        clip_norm=None,
        epochs=1,
        n_samples=1
    )
    assert model is not None, "Model should not be None when clip_norm is None"


def test_invalid_hidden_units():
    """Test that invalid value for hidden_units raises the correct exception."""
    # Empty list should raise an exception
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, hidden_units=[])

    # Non-positive integers should raise an exception
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, hidden_units=[0, 16])

    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, hidden_units=[-10, 16])

    # Non-integer values should raise an exception
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X_IRIS, Y_IRIS_BINARY, hidden_units=[16.5, 8])

    # None is valid (auto-determination)
    model = bayesian_neural_network_classifier_train(
        X_IRIS, 
        Y_IRIS_BINARY, 
        hidden_units=None,
        epochs=1,
        n_samples=1,
    )
    assert model is not None, "Model should not be None when hidden_units is None"
