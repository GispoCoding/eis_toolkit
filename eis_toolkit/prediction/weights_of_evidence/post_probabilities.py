import math
import numpy as np
import rasterio
from typing import Tuple, List


def prior_odds(
    dep_rst: rasterio.io.DatasetReader
) -> Tuple[float, float]:
    """Calculates the prior odds for the training points per unit cell of the study area.

    Args:
        dep_rst (rasterio.io.DatasetReader):  Raster representing the mineral deposits or occurences point data.

    Returns:
        prior_odds_ (float): Prior odds for the mineral deposit per unit cell of the study area.
        inv_dep1s (float): Reciprocal of number of deposit pixels.
    """
    dep_rst_arr = np.array(dep_rst.read(1))
    #dep_size = np.size(dep_rst_arr)
    dep1s = np.count_nonzero(dep_rst_arr == 1)
    dep0s = np.count_nonzero(dep_rst_arr == 0)
    inv_dep1s = 1/dep1s
    prior_probab = dep1s/(dep1s+dep0s)
    prior_odds_ = math.log(prior_probab)/(1-prior_probab)
    return prior_odds_, inv_dep1s


def extract_arrays(
    rasters_gen: List
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates weights summations and variance summations for all evidential rasters

     Args:
         rasters_gen (List): List of raster arrays for all evidential rasters,
         where each element is a 3d array of generalized classes, 
         generalized weights and standard deviation of the corresponding generalized weights. 

    Returns:
         gen_wgts_sum (np.ndarray): Array of sum of generalized weights of all input evidential raster arrays
         var_gen_sum (np.ndarray)]: Array of sum of generalized variance of all input evidential raster arrays

    """
    wgts_gen_ev = [row[1] for row in rasters_gen]
    std_wgts_gen = [row[2] for row in rasters_gen]
    gen_wgts_sum = np.sum(wgts_gen_ev, axis=0)
    var_gen = np.square(std_wgts_gen)
    var_gen_sum = np.sum(var_gen, axis=0)
    return gen_wgts_sum, var_gen_sum


def pprb(
    gen_wgts_sum: np.ndarray,
    prior_odds_: float
) -> np.ndarray:
    """Calculates the final posterior probabilites of presence of the targeted mineral deposit for the given evidential layers.

    Args:
        gen_wgts_sum (np.ndarray): Array of sum of generalized weights of all input evidential raster arrays
        prior_odds (float): Prior odds for the mineral deposit per unit cell of the study area.

    Returns:
        pprb_array (np.ndarray): Array of posterior probabilites of presence of the targeted mineral deposit for the given evidential layers.

    """
    #e = 2.718281828
    #pprb_array =(e**(gen_wgts_sum + prior_odds))/(1+(e**(gen_wgts_sum + prior_odds)))
    pprb_array = (np.exp(gen_wgts_sum + prior_odds_)) / \
        (1+(np.exp(gen_wgts_sum + prior_odds_)))
    return pprb_array


def pprb_stat(
    inv_dep1s: float,
    pprb_array: np.ndarray,
    var_gen_sum: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the standard deviation and the confidence arrays of the posterior probability calculations.

    Args:
        inv_dep1s (float): Reciprocal of number of deposit pixels.
        pprb_array (np.ndarray): Array of posterior probabilites of presence of the targeted mineral deposit for the given evidential layers.
        var_gen_sum (np.ndarray): Array of sum of generalized variance of all input evidential raster arrays.

    Returns:
        pprb_std (np.ndarray): Standard deviations in the posterior probability calculations because of the deviations in weights of the evidential rasters. 
        pprb_conf(np.ndarray): Confidence of the prospectivity values obtained in the posterior probability array.


    """
    pprb_sqr = np.square(pprb_array)
    pprb_std = np.sqrt((inv_dep1s + var_gen_sum) * pprb_sqr)
    pprb_conf = pprb_array/pprb_std
    return pprb_std, pprb_conf
