import numpy as np
import geopandas as gpd
from numbers import Number
from beartype import beartype
from beartype.typing import Union
from rasterio import profiles, transform
from eis_toolkit.vector_processing.distance_computation import distance_computation

@beartype
def calculate_proximity(geodataframe: gpd.GeoDataFrame, raster_profile: Union[profiles.Profile, dict], maximum_distance: Number) -> np.ndarray:

    """ Interpolates the distance values calculated by the distance_computation function between 0 and 1.
        1 denots the value inside the polygon and 0 at the maximum distance.
        If maximum_distance value is not provided, the program sets this value to the maximum value 
        in the provided distance matrix.
        Uses linear interpolation to calculate the distance from the polygon.

        Args:
            geodataframe: The GeoDataFrame with geometries to determine distance to.
            raster_profile: The raster profile of the raster in which the distances
                            to the nearest geometry are determined.
            max_distance: The maximum distance in the output array.

        Returns:
            A 2D numpy array with the the distance values inverted.
    """

    out_matrix = distance_computation(geodataframe, raster_profile, maximum_distance)

    minimum = np.min(out_matrix)
    difference = maximum_distance - minimum
    out_matrix = maximum_distance - out_matrix
    out_matrix = out_matrix/difference

    return out_matrix

@beartype
def calculate_logarithmic_proximity(geodataframe: gpd.GeoDataFrame, raster_profile: Union[profiles.Profile, dict], maximum_distance: Number) -> np.ndarray:

    """ Logarithmically interpolates the distance values calculated by the distance_computation function between 0 and 1.
        1 denots the value inside the polygon and 0 at the maximum distance.
        If maximum_distance value is not provided, the program sets this value to the maximum value 
        in the provided distance matrix.
        Uses linear interpolation to calculate the distance from the polygon.

        Args:
            geodataframe: The GeoDataFrame with geometries to determine distance to.
            raster_profile: The raster profile of the raster in which the distances
                            to the nearest geometry are determined.
            max_distance: The maximum distance in the output array.

        Returns:
            A 2D numpy array with the the distance values inverted.
    """
    
    distance_array = distance_computation(geodataframe, raster_profile,maximum_distance)

    modified_distance_array = np.where(distance_array==0.0,np.nan,distance_array)
    out_matrix = np.log(modified_distance_array)

    log_maximum = np.log(maximum_distance)
    minimum = np.min(distance_array)
    if(minimum != 0):
        log_minimum = np.log(minimum)
    else :
        log_minimum = minimum
    difference = log_maximum - log_minimum
    out_matrix = log_maximum - out_matrix
    out_matrix = out_matrix/difference

    out_matrix = np.nan_to_num(out_matrix,nan=log_maximum)

    return out_matrix