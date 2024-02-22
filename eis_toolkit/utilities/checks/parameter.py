from numbers import Number

from beartype import beartype
from beartype.typing import Any, Sequence, Union


def check_parameter_value(parameter_value: Union[Number, str], allowed_values: Union[list, tuple]) -> bool:
    """
    Check if used parameter value is valid.

    Args:
        parameter_value: value given to a function.
        allowed_values: sequence of values containing allowed parameter values.

    Returns:
        Bool: True if parameter value is allowed, False if not.
    """
    return parameter_value in allowed_values


@beartype
def check_numeric_value_sign(parameter_value: Number) -> bool:
    """Check if input numeric value is positive.

    Args:
        parameter_value: numeric input parameter.

    Returns:
        Bool: True if parameter value is positive, False if not.
    """
    return True if parameter_value > 0 else False


def check_dtype_for_int(scalar: Number) -> bool:
    """
    Determine whether a floating scalar can be converted to integer type.

    Args:
        scalar: Input scalar value.

    Returns:
        True if conversion can be done, False if not.
    """
    return True if isinstance(scalar, int) else scalar.is_integer()


def check_parameter_length(selection: Sequence[int], parameter: Sequence[Any]) -> bool:  # type: ignore[no-untyped-def]
    """
    Check the length of a parameter against the length of selected bands.

    Args:
        selection: Selected bands.
        parameter: List containing parameter values.

    Returns:
        Bool: True if conditions are valid, else False.
    """
    return len(parameter) == 1 or len(parameter) == len(selection)


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
def check_lists_overlap(param_1: Sequence[str], param_2: Sequence[str]) -> bool:
    """
    Check if the lists overlap.

    Args:
        param_1: A list-like parameter.
        param_2: A list-like parameter.

    Returns:
        True if any value is found in both lists.
    """
    return not set(param_1).isdisjoint(param_2)
