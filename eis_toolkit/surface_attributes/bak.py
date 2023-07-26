import scipy
import numpy as np

from beartype import beartype
from beartype import Literal
from numbers import Number

from eis_toolkit.exceptions import NonSquarePixelSizeException


def _get_base_parameters_horn(array: np.ndarray, parameter: Literal, cellsize: Number, mode="reflect") -> np.ndarray:
    kernal_p = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernal_q = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if parameter == "p":
        out_array = scipy.ndimage.correlate(array, weights=kernal_p, mode=mode) / (8 * cellsize)
    elif parameter == "q":
        out_array = scipy.ndimage.correlate(array, weights=kernal_q, mode=mode) / (8 * cellsize)

    return out_array


def _get_base_parameters_zevenbergen():
    pass


def _slope():
    pass


def _aspect():
    pass


def _rad_to_degree(array):
    out_array = array * (180 / np.pi)
    out_array[out_array < 0] = -1

    return out_array


def _rad_to_rise(array):
    out_array = np.tan(array) * 100.0
    out_array[out_array < 0] = -1

    return out_array


def _classify_aspect_pi8():
    pass


def _classify_aspect_pi16():
    pass


def _get_general_curv():
    pass


def _get_profile_curv():
    pass


def _get_plan_curv():
    pass


def _get_tangential_curv():
    pass


def _get_longitudinal_curv():
    pass


def _get_crossectional_curv():
    pass


def _get_minimum_curv():
    pass


def _get_maximum_curv():
    pass


def _get_mean_curv():
    pass


def _get_total_curv():
    pass


def _get_rotor_curv():
    pass


# Checks and error raising
## - quadratic pixel size
## - pixel size >= 1

# Other req.
## - nodata handling
## -
