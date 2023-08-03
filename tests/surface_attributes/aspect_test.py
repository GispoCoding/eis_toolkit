from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidRasterBandException, NonSquarePixelSizeException
from eis_toolkit.surface_attributes.aspect import classify_aspect, get_aspect

parent_dir = Path(__file__).parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")


def test_aspect():
    """Test that computation works as intended."""

    with rasterio.open(raster_path_single) as raster:
        # Default settings
        out_array, out_meta = get_aspect(raster, method="Horn81", unit="degrees")

        # Output shapes and types
        assert isinstance(out_array, np.ndarray)
        assert isinstance(out_meta, dict)

        # Output array (nodata in place)
        test_array = raster.read(1)
        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=-9999, shrink=False).mask,
            np.ma.masked_values(test_array, value=raster.nodata, shrink=False).mask,
        )

        # Minimum slope applied
        out_array_min_slope, _ = get_aspect(raster, unit="degrees", min_slope=2)
        assert not np.array_equal(out_array, out_array_min_slope)


def test_classify_aspect():
    """Test that classification works as intended."""

    num_classes = 8

    with rasterio.open(raster_path_single) as raster:
        # Default settings
        out_array, out_meta = get_aspect(raster, method="Horn81", unit="degrees")

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as aspect:
                aspect.write(out_array, 1)

            with rasterio.MemoryFile(memfile.read()) as dataset_memfile:
                mem_raster = dataset_memfile.open()

            out_array_classified, out_mapping, out_meta = classify_aspect(
                raster=mem_raster, unit="degrees", num_classes=num_classes
            )

    # Output shapes and types
    assert isinstance(out_array_classified, np.ndarray)
    assert isinstance(out_mapping, dict)
    assert isinstance(out_meta, dict)

    # Classification results
    assert -1 <= np.min(out_array_classified) <= num_classes
    assert -1 <= np.max(out_array_classified) <= num_classes


def test_aspect_band_selection():
    """Tests that invalid number of raste bands raise correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path_multi) as raster:
            out_array, out_meta = get_aspect(raster)


def test_aspect_nonsquare_pixelsize():
    """Tests that raster with non-squared pixels raise correct exception."""
    with pytest.raises(NonSquarePixelSizeException):
        with rasterio.open(raster_path_nonsquared) as raster:
            out_array, out_meta = get_aspect(raster)
