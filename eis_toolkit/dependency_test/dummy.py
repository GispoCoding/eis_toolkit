def test_function(a: int, b: int) -> int:
    """Tests whether simple function of summing two values together works.

    Args:
        a (int): first input integer
        b (int): second input integer

    Returns:
        int: sum of the input values
    """
    return(a + b)


def test_function2(a: int) -> int:
    """Tests whether simple function of subtracting 5 from input value works.

    Args:
        a (int): input integer

    Returns:
        int: result after the subtraction operation
    """
    return(a - 5)


print(test_function(1, 2))
print(test_function2(23))
