from eis_toolkit.prediction.weights_of_evidence.post_probabilities import prior_odds, extract_arrays, pprb, pprb_stat
import rasterio
from typing import Tuple, List
import numpy as np


def _calculate_responses(
    dep_rst: rasterio.io.DatasetReader,
    rasters_gen: List
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rstr_meta = dep_rst.meta.copy()
    prior_odds_, inv_dep1s = prior_odds(dep_rst)
    gen_wgts_sum, var_gen_sum = extract_arrays(rasters_gen)
    pprb_array = pprb(gen_wgts_sum, prior_odds_)
    pprb_std, pprb_conf = pprb_stat(inv_dep1s, pprb_array, var_gen_sum)
    return pprb_array, pprb_std, pprb_conf, rstr_meta


def calculate_responses(
    dep_rst: rasterio.io.DatasetReader,
    rasters_gen: List
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Calculates the posterior probability for presence of the targeted mineral deposit for the given evidential layers.

    Args:
        dep_rst (rasterio.io.DatasetReader): Raster representing the mineral deposits or occurences point data.
        rasters_gen (List): List of raster arrays for all evidential rasters, 
        where each element is a 3d array of generalized classes, 
        generalized weights and standard deviation of the corresponding generalized weights.

    Returns:
        pprb_array (np.ndarray): Array of posterior probabilites of presence of the targeted mineral deposit for the given evidential layers.
        pprb_std (np.ndarray): Standard deviations in the posterior probability calculations because of the deviations in weights of the evidential rasters. 
        pprb_conf(np.ndarray): Confidence of the prospectivity values obtained in the posterior probability array.
        array_meta (dict): Resulting raster array's metadata (for visualizations and writing the array to raster file).

    Raises:

    """
    pprb_array, pprb_std, pprb_conf, array_meta = _calculate_responses(
        dep_rst, rasters_gen)
    return pprb_array, pprb_std, pprb_conf, array_meta
