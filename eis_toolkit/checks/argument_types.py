import functools
from typing import Any, Callable

from eis_toolkit.exceptions import InvalidArgumentTypeException


def check_argument_types(func: Callable) -> Callable:
    """
    Check that all arguments correspond to the type hints when a function is invoked.

    This decorator should be used for all public functions of EIS Toolkit for input validation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check positional arguments
        for arg, param in zip(args, func.__annotations__.values()):
            if not isinstance(arg, param):
                raise InvalidArgumentTypeException(f"Expected {param} but got {type(arg)}")

        # Check keyword arguments
        for kwarg, value in kwargs.items():
            expected_type = func.__annotations__.get(kwarg)
            if expected_type is not None and not isinstance(value, expected_type):
                raise InvalidArgumentTypeException(f"Expected {expected_type} but got {type(value)} for '{kwarg}'")

        return func(*args, **kwargs)

    return wrapper
