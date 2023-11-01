import numpy as np


# Test output shapes and types
def check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata, test=1):
    """Check transformation output shape, type and nodata position."""
    # Output shapes and types
    assert out_array.shape == (out_meta["count"], raster.height, raster.width)
    assert isinstance(out_array, np.ndarray)
    assert isinstance(out_meta, dict)
    assert isinstance(out_settings, dict)

    # Output array (nodata in correct place)
    test_array = raster.read(list(range(1, out_meta["count"] + 1)))

    if test == 1:
        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=nodata, shrink=False).mask,
            np.ma.masked_values(test_array, value=nodata, shrink=False).mask,
        )
    elif test == 2:
        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=nodata, shrink=False).mask,
            np.logical_or(
                np.ma.masked_values(test_array, value=nodata, shrink=False).mask,
                np.ma.masked_less_equal(test_array, 0).mask,
            ),
        )
