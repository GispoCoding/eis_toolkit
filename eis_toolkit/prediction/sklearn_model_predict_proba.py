import geopandas as gpd
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Union

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _sklearn_model_predict_proba(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    if sklearnMl._estimator_type == "classifier":
        if Xdf.isna().sum().sum() > 0:
            raise InvalidParameterValueException("DataFrame Xdf contains Nodata-values")
        ydf = pd.DataFrame(sklearnMl.predict_proba(Xdf), columns=sklearnMl.classes_)
    else:
        raise InvalidParameterValueException("Model is not a classifier")

    if igdf is not None and fields is not None:
        if len(ydf.index) != len(igdf.index):
            raise InvalidParameterValueException("Xdf and igdf have different number of rows")
        elif len(igdf.columns) > 0:  # zipping of id and geo columns
            ydf = pd.DataFrame(np.column_stack((igdf, ydf)), columns=igdf.columns.to_list() + ydf.columns.to_list())
            gm = list({i for i in fields if fields[i] in ("g")})
            if len(gm) == 1:
                if gm == ["geometry"]:  # if geometry exists DataFrame will be changed to geoDataFrame
                    ydf = gpd.GeoDataFrame(ydf)
    return ydf


# *******************************
@beartype
def sklearn_model_predict_proba(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of Features for prediction
    igdf: Optional[pd.DataFrame] = None,
    fields: Optional[dict] = None,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    """
        Probabitity of the Prediction of the classes for each sample (row, cell, ...).
        In case igdf is given. the id and geometry column will be zipped to the probability result.
    Args:
        - sklearnMl: Existing classifier model to use for calculation of the prediction probability (random forest classifier, logistic regression,... )
        - Xdf ("array-like"): Features (columns) and samples (rows) to use for calculation of the prediction probability
        - igdf ("array-like"), optional): Columns of identification and geoemtries of the raws
        - fields (optinal): If given it will be used to set the geometry for geodataframe
    Returns:
        pandas Dataframe or geoDataFram containg the prediction probability values for Multiclass prediction
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
    if not hasattr(sklearnMl, "feature_names_in_"):
        raise InvalidParameterValueException("Model is not fitted")

    return _sklearn_model_predict_proba(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        igdf=igdf,
        fields=fields,
    )
