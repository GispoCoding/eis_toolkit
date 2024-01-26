from numbers import Number

import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional
from scipy.signal import ricker


@beartype
def _get_kernel_size(sigma: Number, truncate: Number, size: Optional[int]) -> tuple[int, int]:
    """
    Calculate the kernel size and radius based on the given parameters.

    Args:
        sigma: The standard deviation.
        truncate: The truncation factor.
        size: The optional size of the kernel.

    Returns:
        A tuple containing the calculated size and radius of the kernel.
    """
    if size is not None:
        radius = int(size // 2)
    else:
        radius = int(float(truncate) * float(sigma) + 0.5)
        size = int(2 * radius + 1)

    return size, radius


@beartype
def _create_grid(radius: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of coordinates.

    Args:
        radius: The radius of the grid.
        size: The size of the grid.

    Returns:
        A tuple with x and y coordinates of the grid.
    """
    y, x = np.ogrid[-radius : (size - radius), -radius : (size - radius)]  # noqa: E203
    return x, y


@beartype
def _basic_kernel(size: int, shape: Literal["square", "circle"]) -> np.ndarray:
    """
    Generate a basic kernel of a specified size and shape.

    Args:
        size: The size of the kernel.
        shape: The shape of the kernel. Can be either "square" or "circle".

    Returns:
        The generated kernel.
    """
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
    """
    Generate a Gaussian kernel for image denoising.

    Args:
        sigma: Standard deviation.
        truncate: Truncation value.
        size: Size of the kernel. If not provided, it will be calculated based on sigma and truncate.

    Returns:
        The Gaussian kernel.
    """
    size, radius = _get_kernel_size(sigma, truncate, size)

    x, y = _create_grid(radius, size)
    kernel = 1 / (2 * np.pi * sigma**2) * np.exp((x**2 + y**2) / (2 * sigma**2) * -1)
    kernel /= np.max(kernel)

    return kernel


@beartype
def _mexican_hat_kernel(
    sigma: Number, truncate: Number, size: Optional[int], direction: Literal["rectangular", "circular"]
) -> np.ndarray:
    """
    Generate a Mexican Hat kernel for denoising (circular) or edge detection (rectangular).

    Args:
        sigma: The standard deviation of the Gaussian function.
        truncate: The truncation factor for the Gaussian function.
        size: The size of the kernel.
        direction: Shape of the value distribution, either "rectangular" or "circular".

    Returns:
        The Mexican Hat kernel.
    """
    size, radius = _get_kernel_size(sigma, truncate, size)

    if direction == "rectangular":
        ricker_wavelet = ricker(size, sigma)
        kernel = np.outer(ricker_wavelet, ricker_wavelet)
    elif direction == "circular":
        x, y = _create_grid(radius, size)
        kernel = (
            2
            / np.sqrt(3 * sigma)
            * np.pi ** (1 / 4)
            * (1 - (x**2 + y**2) / sigma**2)
            * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        )
    kernel /= np.max(kernel)

    return kernel
