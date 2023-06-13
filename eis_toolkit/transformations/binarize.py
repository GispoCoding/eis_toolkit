import numpy as np
import rasterio

from numbers import Number
from beartype import beartype
from beartype.typing import Optional, Tuple, Sequence

from eis_toolkit.utilities.miscellaneous import expand_and_zip, cast_dtype_to_int
from eis_toolkit.utilities.nodata import mask_nodata, unmask_nodata

from eis_toolkit.exceptions import InvalidRasterBandException, NonMatchinParameterLengthsException
from eis_toolkit.checks.raster import check_raster_bands
from eis_toolkit.checks.parameter import check_parameter_length


def _binarize(in_array: np.ndarray, threshold: Number) -> np.ndarray:  # type: ignore[no-any-unimported]
    out_array = np.where(in_array <= threshold, 0, 1)

    return out_array


@beartype
def binarize(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]] = None,
    thresholds: Sequence[Number] = [],
    nodata: Optional[Number] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Binarize data based on a given threshold.

    Replaces values less or equal threshold with 0.
    Replaces values greater than the threshold with 1.

    Takes one nodata value which will be re-written after transformation.

    If no band/column selection specified, all bands/columns will be used.
    If only one threshold is specified, it will be used for all (selected) bands.

    Args:
        raster:  Data object to be transformed.
        bands: Selection of bands to be transformed.
        thresholds: Threshold values for transformation.
        nodata: Nodata value to be considered.

    Returns:
        out_array: The transformed data.
        out_meta: Updated metadata.
        out_settings (dict): Log of input settings and calculated statistics if available.

    Raises:
        InvalidRasterBandException: The input contains invalid band numbers.
        NonMatchinParameterLengthsException: The input does not match the number of selected bands
    """
    bands = list(range(1, raster.count + 1)) if bands is None else bands
    nodata = cast_dtype_to_int(raster.nodata if nodata is None else nodata)

    if check_raster_bands(raster, bands) is False:
        raise InvalidRasterBandException("Invalid band selection")

    if not check_parameter_length(bands, thresholds):
        raise NonMatchinParameterLengthsException("Invalid threshold length")

    expanded_args = expand_and_zip(bands, thresholds)
    thresholds = [element[1] for element in expanded_args]

    out_settings = {}

    for i in range(0, len(bands)):
        band_array = raster.read(bands[i])

        band_mask = mask_nodata(band_array, nodata)
        band_array = _binarize(in_array=band_array, threshold=thresholds[i])
        band_array = unmask_nodata(band_array, band_mask, nodata)

        band_array = band_array.astype(np.min_scalar_type(nodata))
        band_array = np.expand_dims(band_array, axis=0)

        out_array = band_array.copy() if i == 0 else np.vstack((out_array, band_array))

        current_transform = f"transformation {i + 1}"
        current_settings = {
            "band_origin": bands[i],
            "threshold": thresholds[i],
            "nodata": nodata,
        }

        out_settings[current_transform] = current_settings

    out_meta = raster.meta.copy()
    out_meta.update({"count": len(bands), "nodata": nodata, "dtype": out_array.dtype.name})

    return out_array, out_meta, out_settings
