from contextlib import contextmanager

from osgeo import gdal


@contextmanager
def toggle_gdal_exceptions():
    """Toggle GDAL exceptions using a context manager.

    If the exceptions are already enabled, this function will do nothing.
    """
    already_has_exceptions_enabled = False
    try:
        if gdal.GetUseExceptions() != 0:
            already_has_exceptions_enabled = True
        gdal.UseExceptions()
        yield
    finally:
        if not already_has_exceptions_enabled:
            gdal.DontUseExceptions()
