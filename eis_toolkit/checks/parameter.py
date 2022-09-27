def check_parameter_value(parameter_value: int, allowed_values: list):
    """Checks if used parameter value is valid.

    Args:
        parameter_value (int): value given to a function.
        allowed_values (list): a list of allowed parameter values.

    Returns:
        Bool: True if parameter value is allowed, False if not
    """

    if parameter_value in allowed_values:
        return True
    else:
        return False


def check_numeric_value_sign(parameter_value):
    """Checks if input numeric value is positive.

    Args:
        parameter value: numeric input parameter

    Returns:
        Bool: True if parameter value is positive, False if not
    """

    if parameter_value > 0:
        return True
    else:
        return False