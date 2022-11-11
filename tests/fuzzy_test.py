import pytest
import numpy
from eis_toolkit.modeling import fuzzy_overlay
from eis_toolkit.exceptions import InvalidParameterValueException
input_data_with_error = numpy.array([[1,-0.5,0.2],[0.9,0.55,0.15],[0.98,0.45,0.3]])
input_data_without_error = numpy.array([[1,0.5,0.2],[0.9,0.55,0.15],[0.98,0.45,0.3]])
input_data_without_error = numpy.array([[1,0.5,NA],[0.9,0.55,0.15],[0.98,0.45,0.3]])
gam_with_error=[-1,2,NA]
gam_without_error=1

def test_fuzzy_overlay_or_invalid_parameter_data():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_or(input_data_with_error)
def test_fuzzy_overlay_and_invalid_parameter_data():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_and(input_data_with_error)
def test_fuzzy_overlay_sum_invalid_parameter_data():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_sum(input_data_with_error)
def test_fuzzy_overlay_product_invalid_parameter_data():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_prod(input_data_with_error)
def test_fuzzy_overlay_gamma_invalid_parameter_data():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_gamma(input_data_with_error,gam_without_error)
def test_fuzzy_overlay_gamma_invalid_parameter_gamma():
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay.fuzzy_gamma(input_data_without_error,gam_with_error)
def 