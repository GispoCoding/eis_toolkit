
# sklearn_randomforest_regressor_test.py
##################################

from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
import pytest
import sys
scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)
from eis_toolkit.prediction.sklearn_randomforest_regressor import *

#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

def test_sklearn_randomforest_regressor():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_randomforest_regressor(oob_score = True) 


def test_sklearn_randomforest_regressor_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(oob_score = 0)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(criterion = 'ich')
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(n_estimators = 10.1)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(max_depth = ['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(max_features = ['1'])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearnML = sklearn_randomforest_regressor(bootstrap = 0)

test_sklearn_randomforest_regressor()
test_sklearn_randomforest_regressor_wrong()

