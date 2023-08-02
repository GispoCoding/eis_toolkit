import geopandas as gpd
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Union

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _sklearn_model_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    cn = ["result"]
    ydf = sklearnMl.predict(Xdf)

    ydf = pd.DataFrame(ydf, columns=cn)
    if igdf is not None:
        if len(ydf.index) != len(igdf.index):
            raise InvalidParameterValueException("Xdf and igdf have different number of rows")
        elif len(igdf.columns) > 0:  # zipping og id and geo columns
            ydf = pd.DataFrame(np.column_stack((igdf, ydf)), columns=igdf.columns.to_list() + ydf.columns.to_list())
            gm = list({i for i in fields if fields[i] in ("g")})
            if len(gm) == 1:  # if geometry exists DataFrame will be changed to geoDataFrame
                if gm == ["geometry"]:
                    ydf = gpd.GeoDataFrame(ydf)

    return ydf


# *******************************
@beartype
def sklearn_model_prediction(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of Features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    """
        Prediction based on a SKLEARN model and  of a DataFrame of samples.
        If given the result will be zipped to Id and geometry columns.
    Args:
        sklearnMl: Existing model to use for the prediction (random rorest classifier, random forest regressor, logistic regression).
        Xdf ("array-like")): Features (columns) and samples (raws) of samples to predict y.
        igdf ("array-like"), optional): Columns of ids and geoemtries.
        fields ( optinal): If given it will be used to set the geometry for geodataframe.
    Returns:
        pandas dataframe or geodataframe containing predicted values. (if geodataframe: geoemtry columns are in the geodataframe)
    """

    # Argument evaluation
    t = sklearnMl.__class__.__name__
    if not t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "Argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException("DataFrame has no column")
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException("DataFrame has no rows")
    if Xdf.isna().sum().sum() > 0:
        raise InvalidParameterValueException("DataFrame Xdf contains Nodata-values")
    if not hasattr(sklearnMl, "feature_names_in_"):
        raise InvalidParameterValueException("Model is not fitted")

    return _sklearn_model_prediction(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        igdf=igdf,
        fields=fields,
    )
