import numpy as np

from eis_toolkit.utilities.conversions import (
    _convert_rad_to_rise,
    convert_deg_to_rad,
    convert_deg_to_rise,
    convert_rad_to_deg,
    convert_rise_to_degree,
)


def test_rad_to_degree():
    """Test that converting radians to degree works as expected."""
    rad = np.arange(0, 2 * np.pi + np.pi / 8, np.pi / 4)

    conversion = convert_rad_to_deg(rad)
    expected = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])

    assert np.array_equal(conversion, expected)


def test_rad_to_precent_rise():
    """Test that converting radians to percent rise works as expected."""
    rad = np.arange(0, np.pi / 2, np.pi / 8)

    conversion = np.round(_convert_rad_to_rise(rad), 3)
    expected = np.array([0, 41.421, 100, 241.421])

    assert np.array_equal(conversion, expected)


def test_deg_to_rad():
    """Test that converting degree to radians works as expected."""
    deg = np.arange(0, 360 + 45, 45)

    conversion = np.round(convert_deg_to_rad(deg), 3)
    expected = np.round(np.arange(0, 2 * np.pi + np.pi / 4, np.pi / 4), 3)

    assert np.array_equal(conversion, expected)


def test_deg_to_precent_rise():
    """Test that converting degrees to percent rise works as expected."""
    deg = np.array([0, 22.5, 45, 67.5])

    conversion = np.round(convert_deg_to_rise(deg), 3)
    expected = np.array([0, 41.421, 100, 241.421])

    assert np.array_equal(conversion, expected)


def test_precent_rise_to_degree():
    """Test that converting degrees to percent rise works as expected."""
    rise = np.tan(np.arange(0, np.pi / 2, np.pi / 8)) * 100

    conversion = np.round(convert_rise_to_degree(rise), 3)
    expected = np.array([0, 22.5, 45, 67.5])

    assert np.array_equal(conversion, expected)
