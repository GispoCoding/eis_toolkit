
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
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(penalty = 'BT')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(dual = 0)
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(tol = 'Ä')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(C = ('Ä'))
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(fit_intercept= 9)
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(class_weight = 9)

        sklearnML = sklearn_logistic_regression(solver = 'Ä')
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(verbose= ('Ä'))
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(warm_start= 9)
    with pytest.raises(InvalidParameterValueException):
        sklearnML = sklearn_logistic_regression(n_jobs = 'P')       

test_sklearn_logistic_regression()
test_sklearn_logistic_regression_wrong()

