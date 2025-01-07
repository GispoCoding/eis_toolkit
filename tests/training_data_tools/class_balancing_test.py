import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import make_classification

from eis_toolkit.exceptions import NonMatchingParameterLengthsException
from eis_toolkit.training_data_tools.class_balancing import balance_SMOTETomek

# CREATE TEST DATA
X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=1,
    n_samples=1000,
    random_state=10,
)


def test_SMOTETomek():
    """Test that balance_SMOTETomek function works as expected."""
    assert not np.array_equal(np.count_nonzero(y == 0), np.count_nonzero(y == 1))  # Class imbalance before balancing

    X_res, y_res = balance_SMOTETomek(X, y)

    np.testing.assert_equal(len(X_res), len(y_res))
    np.testing.assert_equal(np.count_nonzero(y_res == 0), np.count_nonzero(y_res == 1))  # Class balance after balancing


def test_invalid_label_length():
    """Test that different length for feature matrix and labels raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        balance_SMOTETomek(X, np.append(y, "C"))


def test_invalid_sampling_strategy():
    """Test that invalid value for sampling strategy raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        balance_SMOTETomek(X, y, sampling_strategy="invalid_strategy")
