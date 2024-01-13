import numpy as np
import rasterio
from scipy.ndimage import generic_filter, correlate
from scipy.signal import ricker
from numbers import Number
from beartype import beartype
from beartype.typing import Tuple, Literal, Callable, Optional

import sys
sys.path.append('S:/Projekte/20210024_HorizonEurope_GTK_EIS/Bearbeitung/GitHub/eis_toolkit/')
from eis_toolkit.utilities.miscellaneous import reduce_ndim, cast_array_to_float
from eis_toolkit.exceptions import InvalidRasterBandException, InvalidParameterValueException


@beartype
def _check_bands(raster: rasterio.io.DatasetReader):
  if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")
  
  
@beartype
def _check_filter_size(sigma: Optional[Number], truncate: Optional[Number], size: Optional[int]):
  if size is None:
    _, radius = _get_kernel_size(sigma, truncate, size)
    
    if radius < 1:
      raise InvalidParameterValueException("Resulting filter radius too small. Either increase sigma or decrease truncate values.")
  else:
    if size < 3:  
      raise InvalidParameterValueException("Only numbers larger or equal than 3 are allowed for filter size.")
    elif size % 2 == 0:
      raise InvalidParameterValueException("Only odd numbers are allowed for filter size.")
 
  
@beartype
def _check_inputs(raster: rasterio.io.DatasetReader, size: Optional[int], sigma: Optional[Number], truncate: Optional[Number]):
  _check_bands(raster)
  _check_filter_size(sigma, truncate, size)
  
  
@beartype
def _get_kernel_size(sigma: Number, truncate: Number, size: Optional[int]) -> Tuple[int, int]:
  if size is not None:
    radius = int(size // 2)
  else:
    radius = int(float(truncate) * float(sigma) + 0.5)
    size = int(2 * radius + 1)
  
  return size, radius
    
    
@beartype
def _create_grid(radius: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
  y, x = np.ogrid[-radius:size - radius, -radius:size - radius]
  return x, y


@beartype
def _basic_kernel(size: int, shape: Literal["square", "circle"]) -> np.ndarray:
  if shape == "square":
    kernel = np.ones((size, size))
  elif shape == "circle":
    radius = int(size // 2)
    x, y = _create_grid(radius, size)  
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size))
    kernel[mask] = 1
  
  return kernel


@beartype
def _gaussian_kernel(sigma: Number, truncate: Number, size: Optional[int]) -> np.ndarray:  
  size, radius = _get_kernel_size(sigma, truncate, size)
    
  x, y = _create_grid(radius, size)
  kernel = 1 / (2 * np.pi * sigma**2) * np.exp((x**2 + y**2) / (2 * sigma**2) * -1)
  kernel /= np.max(kernel)
  
  return kernel


@beartype
def _mexican_hat_kernel(sigma: Number, truncate: Number, size: Optional[int], direction: Literal["orthogonal", "circular"]) -> np.ndarray:
  size, radius = _get_kernel_size(sigma, truncate, size)
    
  if direction == "orthogonal":
    ricker_wavelet = ricker(size, sigma)
    kernel = np.outer(ricker_wavelet, ricker_wavelet)
  elif direction == "circular":  
    x, y = _create_grid(radius, size)
    kernel = (2 / np.sqrt(3 * sigma) * np.pi**(1/4) * (1 - (x**2 + y**2) / sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2)))
    kernel /= np.max(kernel)
  
  return kernel


@beartype
def _lee_additive_noise(window: np.ndarray, add_noise_var: Number) -> Number:
  p_center = window[window.shape[0] // 2]
  
  if not np.isnan(p_center):  
    local_var = np.nanvar(window)
    local_mean = np.nanmean(window)
    
    weight = local_var / (local_var + add_noise_var)
    return local_mean + weight * (p_center - local_mean)
  else:
    return np.nan
  
  
@beartype
def _lee_multicative_noise(array: np.ndarray, mult_noise_mean: Number, n_looks: int) -> Number:
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
def _lee_additive_multicative_noise(array: np.ndarray, add_noise_var: Number, add_noise_mean: Number, mult_noise_mean: Number) -> Number:
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
def _lee_enhanced(array: np.ndarray, n_looks:int, damping_factor: Number) -> Number:
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
def _lee_sigma() -> Number:
  pass


@beartype
def _frost(array: np.ndarray, damping_factor: int) -> Number:
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
def _kuan(array: np.ndarray, n_looks: int) -> Number:
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
def _apply_generic_filter(array: np.ndarray, filter_fn: Callable, kernel: np.ndarray, *args) -> np.ndarray:
  return generic_filter(array, filter_fn, footprint=kernel, extra_arguments=args)


@beartype
def _apply_correlated_filter(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  return correlate(array, kernel) / np.sum(kernel)


@beartype
def focal_filter(raster: rasterio.io.DatasetReader, 
         method: Literal["mean", "median"] = "mean", 
         size: int = 3, 
         shape: Literal["square", "circle"] = "circle",
         ) -> np.ndarray:
  """
  Apply a basic focal filter to the input raster.

  Parameters:
    raster: The input raster dataset.
    method: The method to use for filtering. Can be either "mean" or "median". Default to "mean".
    size: The size of the filter window. E.g., 3 means a 3x3 window. Default to 3.
    shape: The shape of the filter window. Can be either "square" or "circle". Default to "circle".

  Returns:
    np.ndarray: The filtered raster array.

  Raises:
    InvalidRasterBandException: If the input raster has more than one band.
    InvalidParameterValueException: If the filter size is smaller than 3.
                                    If the filter size is not an odd number.
                                    If the shape is not "square" or "circle".
  """
  _check_inputs(raster, size)
  
  raster_array = raster.read()
  raster_array = reduce_ndim(raster_array)
  
  kernel = _basic_kernel(size, shape)
  
  if method == "mean":
    out_array = _apply_correlated_filter(raster_array, kernel)
  elif method == "median":
    out_array = _apply_generic_filter(raster_array, np.nanmedian, kernel)
  
  return cast_array_to_float(out_array, cast_float=True)


@beartype
def gaussian_filter(raster: rasterio.io.DatasetReader,
                    sigma: Number = 1,
                    truncate: Number = 4,
                    size: Optional[int] = None,
                    ) -> np.ndarray:
  """
  Apply a gaussian filter to the input raster.

  Parameters:
    raster: The input raster dataset.
    sigma: The standard deviation of the gaussian kernel.
    truncate: The truncation factor for the gaussian kernel based on the sigma value. 
              Only if size is not given. Default to 4.0.
              E.g., for sigma = 1 and truncate = 4.0, the kernel size is 9x9.
    size: The size of the filter window. E.g., 3 means a 3x3 window. Default to None.

  Returns:
    np.ndarray: The filtered raster array.

  Raises:
    InvalidRasterBandException: If the input raster has more than one band.
    InvalidParameterValueException: If the filter size is smaller than 3.
                                    If the filter size is not an odd number.
                                    If the resulting radius is smaller than 1.
  """
  _check_inputs(raster, size, sigma, truncate)
  
  raster_array = raster.read()
  raster_array = reduce_ndim(raster_array)
  
  kernel = _gaussian_kernel(sigma, truncate, size)
  out_array = _apply_correlated_filter(raster_array, kernel)
  
  return out_array


@beartype
def mexican_hat_filter(raster: rasterio.io.DatasetReader,
                       sigma: Number = 1,
                       truncate: Number = 4,
                       size: Optional[int] = None,
                       direction: Literal["orthogonal", "circular"] = "circular",
                       ) -> np.ndarray:
  """
  Apply a gaussian filter to the input raster.

  Parameters:
    raster: The input raster dataset.
    sigma: The standard deviation of the gaussian kernel.
    truncate: The truncation factor for the gaussian kernel based on the sigma value. 
              Only if size is not given. Default to 4.0.
              E.g., for sigma = 1 and truncate = 4.0, the kernel size is 9x9.
    size: The size of the filter window. E.g., 3 means a 3x3 window. Default to None.
    direction: The direction of calculating the kernel values. 
               Can be either "orthogonal" or "circular". Default to "circular".

  Returns:
    np.ndarray: The filtered raster array.

  Raises:
    InvalidRasterBandException: If the input raster has more than one band.
    InvalidParameterValueException: If the filter size is smaller than 3.
                                    If the filter size is not an odd number.
                                    If the resulting radius is smaller than 1.
  """
  _check_inputs(raster, size, sigma, truncate)
  
  raster_array = raster.read()
  raster_array = reduce_ndim(raster_array)
  
  kernel = _mexican_hat_kernel(sigma, truncate, size, direction)
  out_array = _apply_correlated_filter(raster_array, kernel)
  
  return out_array
  
 
@beartype
def lee_additive_noise(raster: rasterio.io.DatasetReader,
                       size: int = 3,
                       add_noise_var: Number = 0.25,
                       ) -> np.ndarray:
  """
  Apply a Lee filter with additive noise to the input raster.

  Parameters:
    raster: The input raster dataset.
    size = The size of the filter window. E.g., 3 means a 3x3 window. Default to 3.
    add_noise_var: The additive noise variance. Default to 0.25.

  Returns:
    np.ndarray: The filtered raster array.

  Raises:
    InvalidRasterBandException: If the input raster has more than one band.
    InvalidParameterValueException: If the filter size is smaller than 3.
                                    If the filter size is not an odd number.
  """
  _check_inputs(raster, size)
  
  raster_array = raster.read()
  raster_array = reduce_ndim(raster_array)
  
  kernel = np.ones((size, size))
  out_array = _apply_generic_filter(raster_array, _lee_additive_noise, kernel, add_noise_var)
  
  return out_array


@beartype
def lee_multicative_noise(raster: rasterio.io.DatasetReader,
                          size: int = 3,
                          mult_noise_mean: Number = 1,
                          n_looks: int = 1,
                          ) -> np.ndarray:
  """
  Apply a Lee filter with multiplicative noise to the input raster.

  Parameters:
    raster: The input raster dataset.
    size = The size of the filter window. E.g., 3 means a 3x3 window. Default to 3.
    mult_noise_mean: The multiplative noise mean. Default to 1.
    n_looks: Number of looks to estimate the noise variance. Higher values result in higher smoothing. Default to 1.
    

  Returns:
    np.ndarray: The filtered raster array.

  Raises:
    InvalidRasterBandException: If the input raster has more than one band.
    InvalidParameterValueException: If the filter size is smaller than 3.
                                    If the filter size is not an odd number.
  """
  _check_inputs(raster, size)
  
  raster_array = raster.read()
  raster_array = reduce_ndim(raster_array)
  
  kernel = np.ones((size, size))
  out_array = _apply_generic_filter(raster_array, _lee_multicative_noise, kernel, mult_noise_mean, n_looks)
  
  return out_array
  
  
if __name__ == "__main__":
  # create test array
  np.random.seed(42)
  arr = np.random.rand(5, 5)
  print(arr, "\n")
  
  # ---------------------
  # test function calls
  # ---------------------
  
  # test focal filter
  # ---------------------
  # kernel = _basic_kernel(5, "circle")
  # print(kernel, "\n")
  
  # filtered_array = _apply_generic_filter(arr, np.nanmean, kernel)
  # print(filtered_array)
  
  # filtered_array = cast_array_to_float(filtered_array, cast_float=True)
  # print(filtered_array.dtype)
  
  
  # test gaussian filter
  # ---------------------
  # kernel = _gaussian_kernel(sigma=1, truncate=4, size=None)
  # filtered_array = _apply_correlated_filter(arr, kernel)
  # print(filtered_array, "\n")
  # print(_gaussian_kernel(3, 1))
  
  # test lee_additive_noise
  # ---------------------
  # kernel = np.ones((3, 3))
  # filtered_array = _apply_generic_filter(arr, _lee_additive_noise, kernel, 0.25)
  # print(filtered_array, "\n")
  
  # test lee_multiplicative_noise
  # ---------------------
  kernel = np.ones((3, 3))
  filtered_array = _apply_generic_filter(arr, _lee_multicative_noise, kernel, 1, 1)
  print(filtered_array, "\n")



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
# Check with rasters and may add the np.squeeze() function to remove the third dimension
#
# ---------------------
# Notebook examples
# ---------------------
# Functions for visualization
# Creating different kernels with different parameters
# Applying different filters to an examplary raster





