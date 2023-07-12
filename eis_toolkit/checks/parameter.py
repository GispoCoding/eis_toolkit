from numbers import Number
from typing import Any

from beartype import beartype
from beartype.typing import Iterable


@beartype
def check_parameter_value(parameter_value: Any, allowed_values: Iterable) -> bool:
    """Check if used parameter value is valid.

    Args:
        parameter_value: value given to a function.
        allowed_values: a list of allowed parameter values.

    Returns:
        True if parameter value is allowed, False if not
    """
    if parameter_value in allowed_values:
        return True
    else:
        return False


@beartype
def check_numeric_value_sign(parameter_value: Number) -> bool:
    """Check if input numeric value is positive.

    Args:
        parameter value: numeric input parameter

    Returns:
        True if parameter value is positive, False if not
    """
    if parameter_value > 0:  # type: ignore
        return True
    else:
        return False


@beartype
def check_minmax_position(parameter: tuple) -> bool:  # type: ignore[no-untyped-def]
    """Check if parameter maximum value > parameter minimum value.

    Args:
        parameter: Tuple containing parameter values for min and max.
        
    Returns:
        Bool: True if minimum value < maxiumum value, else False.
    """
    return parameter[0] < parameter[1]


@beartype
def check_dtype_for_int(scalar: Number) -> bool:
    """
    Determine whether a floating scalar can be converted to integer type.

    Args:
        scalar: Input scalar value.
        
    Returns:
        True if conversion can be done, False if not.
    """
    return True if isinstance(scalar, int) else scalar.is_integer()
