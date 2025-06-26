"""Tests for the Bayesian Neural Network module.

This module contains tests for the BayesianNeuralNetworkClassifier class and the
bayesian_neural_network_classifier_train function. It tests both the main functionality
and parameter validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import (
    InsufficientClassesException,
    InvalidDataShapeException,
    InvalidParameterValueException,
)
from eis_toolkit.prediction.bayesian_neural_network import bayesian_neural_network_classifier_train

SEED = 42

X, y = make_classification(
    n_samples=150, n_features=4, n_informative=4, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42
)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def test_bayesian_neural_network_classifier_train():
    """Test that bayesian_neural_network_classifier_train function works as expected."""
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
        bayesian_neural_network_classifier_train(X.ravel(), y)


def test_nonmatching_sample_count():
    """Test that differing number of samples for X and y raises the correct exception."""
    extra_row = np.random.rand(1, X.shape[1])
    X_mismatched = np.concatenate((X, extra_row), axis=0)
    with pytest.raises(InvalidDataShapeException):
        bayesian_neural_network_classifier_train(X_mismatched, y)


def test_invalid_learning_rate():
    """Test that invalid value for learning_rate raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, learning_rate=0.0)


def test_invalid_epochs():
    """Test that invalid value for epochs raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, epochs=0)


def test_invalid_batch_size():
    """Test that invalid value for batch_size raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, batch_size=0)


def test_invalid_n_samples():
    """Test that invalid value for n_samples raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, n_samples=0)


def test_invalid_validation_split():
    """Test that invalid value for validation_split raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, validation_split=-0.1)
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, validation_split=1.1)


def test_invalid_early_stopping_patience():
    """Test that invalid value for early_stopping_patience raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, early_stopping_patience=0)


def test_invalid_early_stopping_min_delta():
    """Test that invalid value for early_stopping_patience raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, early_stopping_min_delta=-0.1)


def test_invalid_early_stopping_monitor():
    """Test that invalid value for early_stopping_monitor raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, early_stopping_monitor="invalid_monitor")


def test_invalid_hidden_units():
    """Test that invalid value for hidden_units raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, hidden_units=[0, 16])

    with pytest.raises(InvalidParameterValueException):
        bayesian_neural_network_classifier_train(X, y, hidden_units=[-10, 16])

    model = bayesian_neural_network_classifier_train(X, y, hidden_units=None, epochs=1, n_samples=1)
    assert model is not None, "Model should not be None when hidden_units is None"


def test_reproducibility():
    """Test that the same seed produces reproducible results across multiple runs."""
    # Reset random state
    np.random.seed(SEED)

    model1 = bayesian_neural_network_classifier_train(
        X_train,
        y_train,
        epochs=5,
        n_samples=5,
        random_state=SEED,
    )
    y_proba1, _ = model1.predict_with_uncertainty(X_test)
    y_pred1 = y_proba1 > 0.5
    history_loss1 = model1.history["loss"][-1]

    # Reset random state
    np.random.seed(SEED)

    model2 = bayesian_neural_network_classifier_train(
        X_train,
        y_train,
        epochs=5,
        n_samples=5,
        random_state=SEED,
    )
    y_proba2, _ = model2.predict_with_uncertainty(X_test)
    y_pred2 = y_proba2 > 0.5
    history_loss2 = model2.history["loss"][-1]

    np.testing.assert_array_equal(y_pred1, y_pred2, err_msg="Predictions should be identical with same seed")

    np.testing.assert_allclose(y_proba1, y_proba2, err_msg="Probabilities should be identical with same seed")

    np.testing.assert_allclose(
        history_loss1, history_loss2, atol=1e-3, err_msg="Final loss values should be nearly identical with same seed"
    )


def test_shuffle_parameter():
    """Test that the shuffle parameter works correctly."""
    model_shuffle = bayesian_neural_network_classifier_train(
        X,
        y,
        epochs=1,
        n_samples=1,
        random_state=SEED,
        shuffle=True,
    )
    assert model_shuffle is not None, "Model should not be None when shuffle is True"

    model_no_shuffle = bayesian_neural_network_classifier_train(
        X,
        y,
        epochs=1,
        n_samples=1,
        random_state=SEED,
        shuffle=False,
    )
    assert model_no_shuffle is not None, "Model should not be None when shuffle is False"


def test_stratified_parameter():
    """Test that the stratified parameter works correctly."""
    model_auto = bayesian_neural_network_classifier_train(
        X,
        y,
        epochs=1,
        n_samples=1,
        validation_split=0.2,
        random_state=SEED,
        stratified=None,
    )
    assert model_auto is not None, "Model should not be None when stratified is None"

    model_stratified = bayesian_neural_network_classifier_train(
        X,
        y,
        epochs=1,
        n_samples=1,
        validation_split=0.2,
        random_state=SEED,
        stratified=True,
    )
    assert model_stratified is not None, "Model should not be None when stratified is True"

    model_not_stratified = bayesian_neural_network_classifier_train(
        X,
        y,
        epochs=1,
        n_samples=1,
        validation_split=0.2,
        random_state=SEED,
        stratified=False,
    )
    assert model_not_stratified is not None, "Model should not be None when stratified is False"


def test_insufficient_classes_exception():
    """Test that training with only one class raises InsufficientClassesException."""
    single_class_y = np.zeros_like(y)

    with pytest.raises(InsufficientClassesException):
        bayesian_neural_network_classifier_train(
            X,
            single_class_y,
            epochs=1,
            n_samples=1,
        )
