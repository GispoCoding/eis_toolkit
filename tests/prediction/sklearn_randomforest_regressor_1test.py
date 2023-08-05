
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

# scripts = r'/eis_toolkit'  # /eis_toolkit/conversions'
# sys.path.append(scripts)
# from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.sklearn_randomforest_regressor import sklearn_randomforest_regressor


def test_sklearn_randomforest_regressor():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_randomforest_regressor(oob_score=True)

    t = (sklearnMl.__class__.__name__)
    assert (t in ("RandomForestRegressor"))


def test_sklearn_randomforest_regressor_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(oob_score=0)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(criterion='ich')
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(n_estimators=10.1)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(max_depth=['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(max_features=['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_regressor(bootstrap=0)


test_sklearn_randomforest_regressor()
test_sklearn_randomforest_regressor_wrong()
