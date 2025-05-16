from numbers import Number

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Tuple, Union
from rasterio import profiles

from eis_toolkit.raster_processing.distance_to_anomaly import distance_to_anomaly
from eis_toolkit.transformations.linear import _min_max_scaling


@beartype
def proximity_to_anomaly(
    anomaly_raster_profile: Union[profiles.Profile, dict],
    anomaly_raster_data: np.ndarray,
    threshold_criteria_value: Union[Tuple[Number, Number], Number],
    threshold_criteria: Literal["lower", "higher", "in_between", "outside"],
    max_distance: Number,
    scaling_range: Tuple[Number, Number] = (1, 0),
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Calculate proximity from raster cell to nearest anomaly.

    The criteria for what is anomalous can be defined as a single number and
    criteria text of "higher" or "lower". Alternatively, the definition can be
    a range where values inside (criteria text of "within") or outside are
    marked as anomalous (criteria text of "outside"). If anomaly_raster_profile does
    contain "nodata" key, np.nan is assumed to correspond to nodata values.

    Scales proximity values linearly in the given range. The first number in scale_range
    denotes the value at the anomaly cells, the second at given maximum_distance.

    Args:
        anomaly_raster_profile: The raster profile in which the distances
            to the nearest anomalous value are determined.
        anomaly_raster_data: The raster data in which the distances
            to the nearest anomalous value are determined.
        threshold_criteria_value: Value(s) used to define anomalous.
            If the threshold criteria requires a tuple of values,
            the first value should be the minimum and the second
            the maximum value.
        threshold_criteria: Method to define anomalous.
        max_distance: The maximum distance in the output array beyond which
            proximity is considered 0.
        scaling_range: Min and max values used for scaling the proximity values.
            Defaults to (1, 0).

    Returns:
        A 2D numpy array with the distances to anomalies computed
        and the original anomaly raster profile.
    """
    out_image, anomaly_raster_profile = distance_to_anomaly(
        anomaly_raster_profile, anomaly_raster_data, threshold_criteria_value, threshold_criteria, max_distance
    )
    out_image = _min_max_scaling(out_image, scaling_range)

    return out_image, anomaly_raster_profile
