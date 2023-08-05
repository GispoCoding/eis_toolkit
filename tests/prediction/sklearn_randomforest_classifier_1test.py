
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

# scripts = r'/eis_toolkit'   # /eis_toolkit/conversions'
# sys.path.append(scripts)

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.sklearn_randomforest_classifier import sklearn_randomforest_classifier


def test_sklearn_randomforest_classifier():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_randomforest_classifier(oob_score=True, class_weight={'Hb': 2, 'Gw': 1, 'I': 3})

    t = (sklearnMl.__class__.__name__)
    assert (t in ("RandomForestClassifier"))


def test_sklearn_randomforest_classifier_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(oob_score=0)
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(InvalidParameterValueException):
        sklearn_randomforest_classifier(ccp_alpha=-2.0)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(criterion='ich')
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(n_estimators=10.1)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(max_depth=['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(max_features=['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_randomforest_classifier(class_weight=0)


test_sklearn_randomforest_classifier()
test_sklearn_randomforest_classifier_wrong()
