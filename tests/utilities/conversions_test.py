import numpy as np

from eis_toolkit.utilities.conversions import (
    _convert_rad_to_rise,
    convert_degree_to_rad,
    convert_degree_to_rise,
    convert_rad_to_degree,
)


def test_rad_to_degree():
    """Test that converting radians to degree works as expected."""
    init = np.linspace(0, 2 * np.pi, 16)
    conversion = convert_rad_to_degree(init)

    assert np.array_equal(np.where(init >= 0, init * (180.0 / np.pi), init), conversion)


def test_rad_to_precent_rise():
    """Test that converting radians to percent rise works as expected."""
    init = np.linspace(0, np.pi / 2, 16)
    conversion = _convert_rad_to_rise(init)

    assert np.array_equal(np.where(init >= 0, np.tan(init) * 100.0, init), conversion)


def test_degree_to_rad():
    """Test that converting degree to radians works as expected."""
    init = np.linspace(0, 360, 16)
    conversion = convert_degree_to_rad(init)

    assert np.array_equal(np.where(init >= 0, (init / 180.0) * np.pi, init), conversion)


def test_degree_to_precent_rise():
    """Test that converting degrees to percent rise works as expected."""
    init = np.linspace(0, np.pi / 2, 16)
    conversion = convert_degree_to_rise(init)

    assert np.array_equal(np.where(init >= 0, np.tan(np.radians(init)) * 100.0, init), conversion)
