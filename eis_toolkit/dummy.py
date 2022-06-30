def test_function(a: int, b: int) -> int:
    """Tests whether simple function of summing two values together works.

    Args:
        a: first input integer
        b: second input integer

    Returns:
        int: sum of the input values
    """
    return(a + b)


print(test_function(1, 2))
