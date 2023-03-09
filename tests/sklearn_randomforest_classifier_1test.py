

import pytest
import sys
scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)
from eis_toolkit.model_training.sklearn_randomforest_classifier import *

#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

def test_sklearn_randomforest_classifier():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_randomforest_classifier(oob_score = True) 


def test_sklearn_randomforest_classifier_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(oob_score = 0)
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(n_estimators = 'A')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(criterion = 'BT')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(max_depth = 'd')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(min_samples_leaf = '0')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(max_features = 'BT')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(max_leaf_nodes = '0')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_classifier(bootstrap = 'A')


test_sklearn_randomforest_classifier()
test_sklearn_randomforest_classifier_wrong()

