import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Tuple, Union
from sklearn import metrics
from sklearn.model_selection import train_test_split

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.sklearn_model_prediction import *

# *******************************


@beartype
def _sklearn_model_validations(
    sklearnMl: Any,   # None: just compare ydf and predict_ydf
    Xdf: Optional[pd.DataFrame] = None,  # dataframe of Features for traning (to split in training and test dataset)
    ydf: Optional[
        pd.DataFrame
    ] = None,  # dataframe of known values for training (to split) or known values to compare with test_y
    predict_ydf: Optional[
        pd.DataFrame
    ] = None,  # predicted values to compare with ydf (known values), if given Xdf is not nessesarry
    # fields: Optional[dict] = None,        # 't'-field will be used if ydf is None
    test_size: Optional[Union[int, float]] = None,  # int: number of test-samples, if float: 0<ts<1
    train_size: Optional[Union[int, float]] = None,  # if None: complement size of the test_size
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = None,
    confusion_matrix: Optional[bool] = True,
    comparison: Optional[bool] = False,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None], Any]:

    # configuration:
    maxlines = 10000  # max. number of y ->rows for comparison

    # xternal tet_set for y will b used
    if predict_ydf is not None:
        if ydf.shape[0] != predict_ydf.shape[0]:
            raise InvalidParameterValueException(
                "Known testset (y) and predicted y have not the same number of samples"
            )
        test_y = ydf
        predict_y = predict_ydf
        testtype = "test_dataset"

    else:  # split in test and training datasets
        testtype = "test_split"
        if test_size is not None:
            if test_size != 0:  #  selftest will be performed
                train_X, test_X, train_y, test_y = train_test_split(
                    Xdf, ydf, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle
                )
            else:
                test_y = train_y = ydf
                test_X = train_X = Xdf
                testtype = "self_test"
        elif train_size is not None:
            if train_size != 0:  # if trainsize == 0:   selftest will be performed
                train_X, test_X, train_y, test_y = train_test_split(
                    Xdf, ydf, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle
                )
            else:
                test_y = train_y = ydf
                test_X = train_X = Xdf
                testtype = "self_test"

        #  Training based on the training-data
        ty = train_y
        if len(train_y.shape) > 1:
            if train_y.shape[1] == 1:
                ty = np.ravel(train_y)

        if sklearnMl._estimator_type == "classifier":
            if np.issubdtype(ty.dtype, np.floating):
                raise InvalidParameterValueException("A classifier model cannot use a float y (target)")
        else:
            if not np.issubdtype(ty.dtype, np.number):
                raise InvalidParameterValueException("A regressor model can only use number y (target)")

        sklearnMl.fit(train_X, ty)

        # Prediction based on the test-data
        predict_y = sklearn_model_prediction(sklearnMl, test_X)

    # Validation
    validation = {}
    if sklearnMl._estimator_type == "regressor":
        validation["R2 score"] = metrics.r2_score(test_y, predict_y)
        validation["explained variance"] = metrics.explained_variance_score(test_y, predict_y)
        validation["mean absolut error"] = metrics.mean_absolute_error(test_y, predict_y)
        validation["mean square arror"] = metrics.mean_squared_error(test_y, predict_y)
    else:
        validation["accuracy"] = metrics.accuracy_score(test_y, predict_y)
        validation["recall"] = metrics.recall_score(test_y, predict_y, average="weighted")
        validation["precision"] = metrics.precision_score(test_y, predict_y, average="weighted")
        validation["F1 score"] = metrics.f1_score(test_y, predict_y, average="weighted")
    if hasattr(sklearnMl, "oob_score"):
        if sklearnMl.oob_score:
            validation["oob score"] = sklearnMl.oob_score_
    validation["testsplit size"] = test_y.shape[0]
    validation = pd.DataFrame.from_dict(validation, orient="index", columns=[testtype])

    # confusion matrix
    confusion1 = None
    if sklearnMl._estimator_type == "classifier":
        if confusion_matrix:
            ltest = test_y.loc[:, test_y.columns[0]].tolist()
            lpredict = predict_y.loc[:, predict_y.columns[0]].tolist()
            lists = list(set(ltest + lpredict))
            lists.sort()
            confusion = pd.DataFrame(
                metrics.confusion_matrix(ltest, lpredict)
            )

            list2 = list(confusion.index.values)
            df1 = confusion.rename(index=dict(zip(list2, lists)))
            confusion1 = df1.rename(columns=dict(zip(list2, lists)))

    # comparison
    comparison_lst = None
    if comparison and test_y.shape[0] < maxlines:
        predict_y.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)
        comparison_lst = test_y.join(predict_y)

    return validation, confusion1, comparison_lst, sklearnMl


# *******************************
@beartype
def sklearn_model_validations(
    sklearnMl: Any,
    Xdf: Optional[pd.DataFrame] = None,  # dataframe of Features for splitting in traning and test dataset
    ydf: Optional[pd.DataFrame] = None,  # dataframe of known values for splitting
    predict_ydf: Optional[pd.DataFrame] = None,  # if No random subset will be used for validation (test_size...)
    test_size: Optional[Union[int, float]] = None,  # int: namuber of test-samples, if float: 0<ts<1
    train_size: Optional[Union[int, float]] = None,  # if None: complement size of the test_size
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = None,
    confusion_matrix: Optional[bool] = True,  # calculate confusion matrix
    comparison: Optional[bool] = False,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None], Any]:

    """
       Validation for a ML model based on:
       - random splited testset from size test_size/train_size.
         Xdf and ydf will be randomly splitted in a test and a training dataset.
         The training-dataset will be used for model-training.
         The test-dataset will be used for prediction.
         The result of prediction will be compared with ydf from test-dataset
       - if predict_ydf ist given:  test_ydf is the known set of data to compare with predict_ydf

    Args:
       - sklearnMl (model). The Model will be fitted based on the training dataset.
          Even for comparison with a testset (or verification dataset) the model is used to get the model-typ (regression or classification).
       - Xdf ("array-like"): features (columns) and samples (rows)
       - ydf ("array-like"): target valus(one column) and samples (rows) (same number as Xdf)
            In case the estimator is a classifier ydf should be int.
            If ydf is = None, target column is included in Xdf. In this case "fields" should not be None and one column should be the target ('t').
       - predict_ydf: ("array-like"): predicted values of a test dataset (validation dataset).
       - test_size (default=None):
          If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
          If int, represents the absolute number of test samples.
          If None (or negative), the value is set to the complement of the train size. If train_size is also None (or <0), it will be set to 0.25.
          If = 0 or 0.0: a selftest will be peformed: test-set = train-set
       - train_size (default=None):
          If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
          If int, represents the absolute number of train samples.
          If None or negative, the value is automatically set to the complement of the test size.
          if = 0 or 0.0 a selftest will be performed: : test-set = train-set
       - random_state (default=None):
          Controls the shuffling applied to the data before applying the split.
          Pass an int for reproducible output across multiple function calls.
          No effect with selftest.
       - shuffle (default=True):
          Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
          No effect with selftest
       - confusion_marix (default=True): to create?
       - comparision (default=False): to create?

    Returns:
       DataFrame with all values of the validation
       DataFrame of confusion matrix, if calculated
            confusion schold be read paire wise:
            number of pair test(0),predict(0),
            up to n classes which apeers in y_test and y_predict
       DataFrame for comparison list, if calculated (max 10000 lines will be listed)
       Model, if calculated
    """

    # Argument evaluation
    t = sklearnMl.__class__.__name__

    if not t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "argument sklearnMl is not one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    if predict_ydf is not None:
        if len(predict_ydf.columns) != 1:
            raise InvalidParameterValueException("predict_ydf has 0 or more than 1 columns")
    else:
        if sklearnMl is None:
            raise InvalidParameterValueException("both Sklaern and predict_ydf are None")
    if Xdf is not None:
        if len(Xdf.columns) == 0:
            raise InvalidParameterValueException("DataFrame has no column")
        if len(Xdf.index) == 0:
            raise InvalidParameterValueException("DataFrame has no rows")
    if train_size is not None:
        if train_size < 0:
            raise InvalidParameterValueException("train_size is <0")
    if test_size is not None:
        if test_size < 0:
            raise InvalidParameterValueException("test_size is <0")

    validation, confusion, comparison, sklearnMl = _sklearn_model_validations(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        ydf=ydf,
        predict_ydf=predict_ydf,
        # fields = fields,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        confusion_matrix=confusion_matrix,
        comparison=comparison,
    )

    return validation, confusion, comparison, sklearnMl
