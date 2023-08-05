
import pytest

# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)

from eis_toolkit.prediction.sklearn_logistic_regression import sklearn_logistic_regression


def test_sklearn_logistic_regression():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_logistic_regression()

    t = sklearnMl.__class__.__name__
    # t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
    assert t in ("LogisticRegression")


def test_sklearn_logistic_regression_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(penalty=0)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(dual=-2.0)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(solver="ich")
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(max_iter=10.1)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(verbose=["1"])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(C=["1"])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_logistic_regression(class_weight=0)


test_sklearn_logistic_regression()
test_sklearn_logistic_regression_wrong()
