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


def check_resample_upscale_factor(upscale_factor: float):
    """Checks if upscale factor used in resampling is positive.

    Args:
        upscale_factor (float): resample factor for raster resampling

    Returns:
        Bool: True if resample upscale factor is positive, False if not
    """

    if upscale_factor > 0:
        return True
    else:
        return False