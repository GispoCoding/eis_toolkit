import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.raster_processing.create_constant_raster import create_constant_raster
from eis_toolkit.exceptions import InvalidParameterValueException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_create_from_template():
    """Test that the raster creation from template works as intended"""

    constant = 1
    nodata = 99

    with rasterio.open(raster_path) as raster:
        out_array, out_meta = create_constant_raster(constant, template_raster=raster)

    # Output shapes and types
    assert isinstance(out_array, np.ndarray)
    assert isinstance(out_meta, dict)

    assert out_array.shape == (raster.height, raster.width)
    assert out_meta["crs"] == raster.meta["crs"]
    assert out_meta["transform"] == raster.meta["transform"]
    assert out_meta["nodata"] == raster.meta["nodata"]

    with rasterio.open(raster_path) as raster:
        out_array, out_meta = create_constant_raster(constant, template_raster=raster, nodata_value=nodata)

    assert out_meta["nodata"] == nodata


def test_create_from_origin():
    """Test that the raster creation from origin works as intended"""

    constant = 1
    nodata = 99
    width = 100
    height = 100
    west = 384744.0
    north = 6671384.0
    epsg = 3067
    pixel_size = 20

    out_array, out_meta = create_constant_raster(
        constant_value=constant,
        coord_west=west,
        coord_north=north,
        target_epsg=epsg,
        target_pixel_size=pixel_size,
        raster_width=width,
        raster_height=height,
        nodata_value=nodata,
    )

    # Output shapes and types
    assert isinstance(out_array, np.ndarray)
    assert isinstance(out_meta, dict)

    assert out_array.shape == (height, width)
    assert out_meta["crs"] == rasterio.crs.CRS.from_epsg(epsg)
    assert out_meta["transform"] == rasterio.transform.from_origin(west, north, pixel_size, pixel_size)
    assert out_meta["nodata"] == nodata


def test_create_from_bounds():
    """Test that the raster creation from bounds works as intended"""

    constant = 1
    nodata = 99
    width = 100
    height = 100
    west = 384744.0
    north = 6671384.0
    east = west + width
    south = north - height
    epsg = 3067

    out_array, out_meta = create_constant_raster(
        constant_value=constant,
        coord_west=west,
        coord_north=north,
        coord_east=east,
        coord_south=south,
        target_epsg=epsg,
        raster_width=width,
        raster_height=height,
        nodata_value=nodata,
    )

    # Output shapes and types
    assert isinstance(out_array, np.ndarray)
    assert isinstance(out_meta, dict)

    assert out_array.shape == (height, width)
    assert out_meta["crs"] == rasterio.crs.CRS.from_epsg(epsg)
    assert out_meta["transform"] == rasterio.transform.from_bounds(west, south, east, north, width, height)
    assert out_meta["nodata"] == nodata


def test_constant_raster_shape():
    """Test that invalid height and width values raise the correct exception."""

    with pytest.raises(InvalidParameterValueException):
        create_constant_raster(
            constant_value=1,
            coord_west=0,
            coord_north=0,
            target_epsg=3067,
            target_pixel_size=20,
            raster_width=0,
            raster_height=10,
            nodata_value=-999,
        )

        create_constant_raster(
            constant_value=1,
            coord_west=0,
            coord_north=0,
            target_epsg=3067,
            target_pixel_size=20,
            raster_width=10,
            raster_height=-10,
            nodata_value=-999,
        )


def test_constant_raster_by_bounds_coordinates():
    """Test that invalid coordinate values raise the correct exception."""

    with pytest.raises(InvalidParameterValueException):
        create_constant_raster(
            constant_value=1,
            coord_west=0,
            coord_north=100,
            coord_east=-100,
            coord_south=0,
            target_epsg=3067,
            raster_width=100,
            raster_height=100,
            nodata_value=-999,
        )

        create_constant_raster(
            constant_value=1,
            coord_west=0,
            coord_north=0,
            coord_east=100,
            coord_south=100,
            target_epsg=3067,
            raster_width=100,
            raster_height=100,
            nodata_value=-999,
        )


def test_constant_raster_origin_pixel_size():
    """Test that invalid pixel size raises the correct exception."""

    with pytest.raises(InvalidParameterValueException):
        create_constant_raster(
            constant_value=1,
            coord_west=0,
            coord_north=0,
            target_epsg=3067,
            target_pixel_size=-1,
            raster_width=100,
            raster_height=100,
            nodata_value=-999,
        )
