from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from eis_toolkit.exceptions import InvalidParameterValueException


def _featureclass_to_geopandas(  # type: ignore[no-any-unimported]
    fc: str
) -> gpd.GeoDataFrame:

    dfg = gpd.read_file(fc)
    return dfg


def featureclass_to_geopandas(  # type: ignore[no-any-unimported]
    fc: str
) -> gpd.GeoDataFrame:

    """Convert featureclass to pandas geoDataFrame.

    Args:
        fc (string): name of the featureclass 

    Returns:
        gpd.eoDataFrame
    """
    dfg = _featureclass_to_geopandas(fc)

    return dfg
