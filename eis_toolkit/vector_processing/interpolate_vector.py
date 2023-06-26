from scipy.spatial.distance import cdist
import numpy as np


def interpolate_geodataframe_idw(geodataframe, target_position, power=2):
    """
    Interpolates data from a GeoDataFrame using Inverse Distance Weighting (IDW) based on the given target position.

    Args:
        geodataframe (GeoDataFrame): The GeoDataFrame containing the input data.
        target_position (tuple): The target position for interpolation.
        power (float, optional): The power parameter for IDW. Defaults to 2.

    Returns:
        dict: A dictionary containing the interpolated values for each column in the GeoDataFrame.
    """
    interpolated_values = {}

    # Extract the positions and values from the GeoDataFrame
    positions = geodataframe.geometry.apply(lambda geom: (geom.x, geom.y))
    values = geodataframe.drop('geometry', axis=1)

    # Extract x and y coordinates from positions
    x_coordinates, y_coordinates = zip(*positions)

    # Calculate distances between target position and input positions
    input_positions = list(zip(x_coordinates, y_coordinates))
    distances = cdist([target_position], input_positions)

    # Calculate the weights based on distances using IDW
    weights = 1 / distances**power

    # Normalize the weights
    normalized_weights = weights / np.sum(weights)

    # Interpolate values for each column using the weights
    for column in values.columns:
        interpolated_value = np.sum(normalized_weights * values[column].values)
        interpolated_values[column] = interpolated_value

    return interpolated_values
