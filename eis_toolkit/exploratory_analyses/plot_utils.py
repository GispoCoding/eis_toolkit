import numpy as np


def get_default_palette(data, hue, palette):
    """Get default palette based on data type."""

    # Check if hue column data is numeric
    if np.issubdtype(data[hue].dtype, np.number):
        # if palette is not specified by user, set it to 'viridis'
        return palette if palette else "viridis"
    else:
        # if palette is not specified by user, set it to 'deep'
        return palette if palette else "deep"
