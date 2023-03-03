
# sklearn_randomforest_regressor_test.py
##################################
import pytest
import sys
scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)
from eis_toolkit.model_training.sklearn_randomforest_regressor import *

#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

def test_sklearn_randomforest_regressor():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_randomforest_regressor(oob_score = True) 


def test_sklearn_randomforest_regressor_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_randomforest_regressor(oob_score = 0)

test_sklearn_randomforest_regressor()
test_sklearn_randomforest_regressor_wrong()

