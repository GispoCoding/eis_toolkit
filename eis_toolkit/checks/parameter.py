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
def check_numeric_value_sign(parameter_value: Number) -> bool:  # type: ignore[no-untyped-def]
    """Check if input numeric value is positive.

    Args:
        parameter value: numeric input parameter

    Returns:
        True if parameter value is positive, False if not
    """
    if parameter_value > 0:
        return True
    else:
        return False
