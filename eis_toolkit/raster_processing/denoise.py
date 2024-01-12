import numpy as np
import rasterio
from scipy.ndimage import generic_filter, correlate
from scipy.signal import ricker
from numbers import Number
from beartype import beartype
from beartype.typing import Tuple, Literal

from eis_toolkit.exceptions import InvalidRasterBandException, InvalidParameterValueException


@beartype
def _check_inputs(raster: rasterio.io.DatasetReader, size: int):
  if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

  if size <= 1 or size % 2 == 0:
      raise InvalidParameterValueException("Only odd numbers larger than 1 are allowed for filter size.")


@beartype
def _create_grid(radius: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
  y, x = np.ogrid[-radius:size - radius, -radius:size - radius]
  return x, y


@beartype
def _basic_kernel(size: int, shape: Literal["square", "circle"]) -> np.ndarray:
  if shape == "square":
    kernel = np.ones((size, size))
  elif shape == "circle":
    radius = size // 2
    x, y = _create_grid(radius, size)  
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size))
    kernel[mask] = 1
  
    return kernel
  

@beartype
def _gaussian_kernel(size: int, sigma: Number) -> np.ndarray:  
  radius = size // 2
  x, y = _create_grid(radius, size)
  kernel = 1 / (2 * np.pi * sigma**2) * np.exp((x**2 + y**2) / (2 * sigma**2) * -1)
  kernel /= np.max(kernel)
  
  return kernel
  

@beartype
def _mexican_hat_kernel(size: int, sigma: Number, direction: Literal["orthogonal", "circular"]) -> np.ndarray:
  if direction == "orthogonal":
    ricker_wavelet = ricker(size, sigma)
    kernel = np.outer(ricker_wavelet, ricker_wavelet)
  elif direction == "circular":  
    radius = size // 2
    x, y = _create_grid(radius, size)
    kernel = (2 / np.sqrt(3 * sigma) * np.pi**(1/4) * (1 - (x**2 + y**2) / sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2)))
    kernel /= np.max(kernel)
  
  return kernel


@beartype
def _lee_additive_noise(array: np.ndarray, add_noise_var: int) -> np.ndarray:
  p_center = array[array.shape[0] // 2]
  
  if not np.isnan(p_center):  
    local_var = np.nanvar(array)
    local_mean = np.nanmean(array)
    
    weight = local_var / (local_var + add_noise_var)
    return local_mean + weight * (p_center - local_mean)
  else:
    return np.nan
  
  
@beartype
def _lee_multicative_noise(array: np.ndarray, mult_noise_mean: int, n_looks: int) -> np.ndarray:
  p_center = array[array.shape[0] // 2]
  
  if not np.isnan(p_center):
    local_var = np.nanvar(array)
    local_mean = np.nanmean(array)
    
    mult_noise_var = 1 / n_looks
    weight = mult_noise_mean * local_var / ((mult_noise_var * local_mean**2)+ (local_var * mult_noise_mean**2))
    return local_mean + weight * (p_center - mult_noise_mean * local_mean)
  else:
    return np.nan


@beartype
def _lee_additive_multicative_noise(array: np.ndarray, add_noise_var: int, add_noise_mean: int, mult_noise_mean: int) -> np.ndarray:
  p_center = array[array.shape[0] // 2]  
  
  if not np.isnan(p_center):
    local_var = np.nanvar(array)
    local_sd = np.nanstd(array)
    local_mean = np.nanmean(array)  
    
    mult_noise_var = np.power((local_sd / local_mean), 2)
    weight = mult_noise_mean * local_var / ((mult_noise_var * local_mean**2)+ (local_var * mult_noise_mean**2) + add_noise_var)
    return local_mean + weight * (p_center - mult_noise_mean * local_mean - add_noise_mean)
  else:
    return np.nan


@beartype
def _lee_enhanced(array: np.ndarray, n_looks:int, damping_factor: Number) -> np.ndarray:
  p_center = array[array.shape[0] // 2]
  
  if not np.isnan(p_center):  
    local_sd = np.nanstd(array)
    local_mean = np.nanmean(array)
    
    c_u = 1 / np.sqrt(n_looks)
    c_max = np.sqrt(1 + 2 / n_looks)
    c_i = local_sd / local_mean
    
    exponent = -damping_factor * (c_i - c_u) / (c_max - c_i)
    weight = np.exp(exponent)
    
    if c_i <= c_u:
      weighted_value = local_mean
    elif c_u < c_i and c_i < c_max:
      weighted_value = local_mean * weight + p_center * (1 - weight)
    elif c_i >= c_max:
      weighted_value = p_center
      
    return weighted_value
  else:
    return np.nan


@beartype
def _lee_sigma() -> np.ndarray:
  pass


@beartype
def _frost(array: np.ndarray, damping_factor: int) -> np.ndarray:
  p_center = array[array.shape[0] // 2]
  
  if not np.isnan(p_center):
    s_dist = np.abs(array - p_center)
    local_var = np.nanvar(array)
    local_mean = np.nanmean(array)
    
    scaled_var = local_var / local_mean**2
    factor_b = damping_factor * scaled_var
    array_weights = np.exp(-factor_b * s_dist)
            
    weighted_array = array * array_weights
    return np.nansum(weighted_array) / np.nansum(array_weights)
  else:
    return np.nan
    
    
@beartype
def _kuan(array: np.ndarray, n_looks: int) -> np.ndarray:
  p_center = array[array.shape[0] // 2]
  
  if not np.isnan(p_center):
    local_sd = np.nansd(array)
    local_mean = np.nanmean(array)
    
    c_u = 1 / np.sqrt(n_looks)
    c_i = local_sd / local_mean
    
    weight = (1 - (c_u**2 / c_i**2)) / (1 + c_u**2)
    return p_center * weight + local_mean * (1 - weight)
  else:
    return np.nan


@beartype
def _focal_filter(array: np.ndarray, filter_fn: callable, kernel: np.ndarray) -> np.ndarray:
  return generic_filter(array, filter_fn, footprint=kernel)

def focal_filter(raster: rasterio.io.DatasetReader, method: Literal["mean", "median"], size: int, shape: Literal["square", "circle"]) -> np.ndarray:
  _check_inputs(raster, size)
  array = raster.read(1)
  kernel = _basic_kernel(size, shape)
  
  if method == "mean":
    filter_fn = np.nanmean
  elif method == "median":
    filter_fn = np.nanmedian
    
  out_array = generic_filter(arr, filter_fn, footprint=kernel)
  return out_array
 
  
if __name__ == "__main__":
  # create test array
  np.random.seed(42)
  arr = np.random.rand(200, 200)
  
  # test function calls
  # print(_gaussian_kernel(3, 1))



###### Coding list
# --------------------
# Filter
# ---------------------
#
# Focal mean
# Focal Median
# Gaussian
# Mexican Hat
# Lee
# Lee Enhanced
# Lee Sigma
# Kuan
# Frost
#
# ---------------------
# General Points
# ---------------------
# Circle and rectangular shaped kernels
#
# ---------------------
# Exceptions
# ---------------------
# Only single band raster
# No negative values for specific parameters
# Filter size not larger than raster dimensions
# Only odd numbers are allowed for filter size
# NLooks > 0
#
# ---------------------
# References
# ---------------------
# ArcGIS Forumulas for Speckle filters
# https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm
#
# ---------------------
# Misc. Notes
# ---------------------
# ...
#
# ---------------------
# Notebook examples
# ---------------------
# Functions for visualization
# Creating different kernels with different parameters
# Applying different filters to an examplary raster





