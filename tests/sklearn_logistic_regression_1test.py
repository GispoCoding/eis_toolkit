
# sklearn_logistic_regression_test.py
##################################
import pytest
import sys
scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)
from eis_toolkit.model_training.sklearn_logistic_regression import *

#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

def test_sklearn_logistic_regression():
    """Test functionality of creating a model."""
    sklearnMl = sklearn_logistic_regression() 


def test_sklearn_logistic_regression_wrong():
    """Test functionality of creating a model with wrong arguments."""
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(penalty = 0)

test_sklearn_logistic_regression()
test_sklearn_logistic_regression_wrong()

