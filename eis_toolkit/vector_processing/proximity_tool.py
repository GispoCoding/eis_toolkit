import numpy as np
from beartype import beartype
from beartype.typing import  Optional
from numbers import Number

@beartype
def calculate_proximity(distance_array: np.ndarray, maximum_distance: Optional[Number] = None) -> np.ndarray:

    """ Interpolates the distance values calculated by the distance_computation function between 0 and 1.
        1 denots the value inside the polygon and 0 at the maximum distance.
        If maximum_distance value is not provided, the program sets this value to the maximum value 
        in the provided distance matrix.
        Uses linear interpolation to calculate the distance from the polygon.

        Args:
            distance_array: The distance array calculated from the distance_computation
            maximum_distance: The maximum distance from the polygon

        Returns:
            A 2D numpy array with the the distance values inverted.
    """
    
    if (maximum_distance is None):
        maximum_distance = np.max(distance_array)
        out_matrix = distance_array
    else :
        out_matrix = np.where(distance_array>maximum_distance,maximum_distance,distance_array)

    minimum = np.min(distance_array)
    difference = maximum_distance - minimum
    out_matrix = maximum_distance - out_matrix
    out_matrix = out_matrix/difference

    return out_matrix