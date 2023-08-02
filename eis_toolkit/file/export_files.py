import os

import pathlib

import joblib
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional, Union

from eis_toolkit.exceptions import FileWriteError, InvalidParameterValueException


# *******************************
@beartype
def _export_files(
    name: Optional[str] = "ML",  # Name of Subject
    path: Optional[Union[str, pathlib.PosixPath]] = None,
    validations: Optional[pd.DataFrame] = None,
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    sklearnMl: Optional[Any] = None,
    sklearnOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False,
) -> dict:
    def create_file(path: str, name: str, extension: str, new_version: bool) -> str:
        if len(extension) > 0:
            extension = "." + extension
        filenum = 1
        filename = os.path.join(path, name)
        if os.path.exists(os.path.abspath(filename + extension)):
            if new_version:  # next file number
                while os.path.exists(os.path.abspath(filename + str(filenum) + extension)):
                    filenum += 1
                return filename + str(filenum)
            else:  # file will be deletet
                try:
                    os.remove(os.path.abspath(filename + extension))
                except:
                    raise FileWriteError("Problem deleting file " + str(filename + extension))
        return filename

    # Main
    dt = {}  # dictionary of output files to use in input_files function
    if path is None:
        parent_dir = pathlib.Path(__file__).parent.parent
        path = parent_dir.joinpath(r"data")
    if not os.path.exists(path):
        raise FileWriteError("path does not exists:" + str(path))
    dt["path"] = path

    if decimalpoint_german:
        decimal = ","
        separator = ";"
    else:
        decimal = "."
        separator = ","

    if validations is not None:  # Validation
        # to csv
        file = create_file(path, name + "_validation", "csv", new_version=new_version)
        try:
            validations.to_csv(file + ".csv", sep=separator, header=True, decimal=decimal, float_format="%00.5f")
        except:
            raise FileWriteError("Problem saving file " + file + ".csv")
        file = create_file(path, name + "_validation", "json", new_version=new_version)
        try:
            validations.to_json(file + ".json", orient="split", indent=3)
        except:
            raise FileWriteError("Problem saving file " + file + ".json")

    if validations is not None and confusion_matrix is not None:
        # to csv
        file = create_file(path, name + "_confusion_matrix", "csv", new_version=new_version)
        try:
            confusion_matrix.to_csv(
                file + ".csv", sep=separator, index=True, header=True, decimal=decimal
            )
        except:
            raise FileWriteError("Problem saving file " + file + ".csv")
        # to json
        file = create_file(path, name + "_confusion_matrix", "json", new_version=new_version)
        try:
            confusion_matrix.to_json(file + ".json", double_precision=5, orient="table", indent=3)  # to string
        except:
            raise FileWriteError("Problem saving file " + file + ".json")

    if crossvalidation is not None:  # cross validation
        # inner arrays to dictionaries
        ndict = {}
        for i in crossvalidation:  # i: keys
            nfolds = {}
            zl = 0
            for j in crossvalidation[i]:  # array of the folds
                nfolds["fold" + (zl + 1).__str__()] = j
                zl += 1
            ndict[i] = nfolds

        # to csv
        file = create_file(path, name + "_cross_validation", "csv", new_version=new_version)
        tmp = pd.DataFrame.from_dict(ndict, orient="index")  # über dataframe

        try:
            tmp.to_csv(
                file + ".csv", sep=";", index=True, header=True, float_format="%00.5f", decimal=decimal
            )  # formatstring
        except:
            raise FileWriteError("Problem saving file " + file + ".csv")

        file = create_file(path, name + "_cross_validation", "json", new_version=new_version)
        df = pd.DataFrame.from_dict(ndict)
        try:
            df.to_json(file + ".json", double_precision=5, orient="table", indent=3)  # to string
        except:
            raise FileWriteError("Problem saving file " + file + ".json")

    if comparison is not None:
        # to csv
        file = create_file(path, name + "_comparison", "csv", new_version=new_version)
        try:
            comparison.to_csv(
                file + ".csv", sep=separator, index=True, header=True, decimal=decimal
            )
        except:
            raise FileWriteError("Problem saving file " + file + ".csv")
        # to json
        file = create_file(path, name + "_comparison", "json", new_version=new_version)
        try:
            comparison.to_json(file + ".json", double_precision=5, orient="table", indent=3)  # to string
        except:
            raise FileWriteError("Problem saving file " + file + ".json")
    if importance is not None:
        # to csv
        file = create_file(path, name + "_importance", "csv", new_version=new_version)
        try:
            importance.to_csv(
                file + ".csv", sep=separator, index=True, header=True, decimal=decimal
            )
        except:
            raise FileWriteError("Problem saving file " + file + ".csv")
        # to json
        file = create_file(path, name + "_importance", "json", new_version=new_version)
        try:
            importance.to_json(file + ".json", double_precision=5, orient="table", indent=3)  # to string
        except:
            raise FileWriteError("Problem saving file " + file + ".json")

    if sklearnMl is not None:  # Model
        file = create_file(path, name + "_sklearnMl", "mdl", new_version=new_version)
        dt["sklearnMl"] = file + ".mdl"
        try:
            joblib.dump(sklearnMl, file + ".mdl")  # save the model
        except:
            raise FileWriteError("Problem saving file " + file + ".mdl")

    if sklearnOhe is not None:  # OneHotEncoder
        file = create_file(path, name + "_sklearnOhe", "ohe", new_version=new_version)
        dt["sklearnOhe"] = file + ".ohe"
        try:
            joblib.dump(sklearnOhe, file + ".ohe")
        except:
            raise FileWriteError("Problem saving file " + file + ".ohe")
    # fields
    if myFields is not None:  # Validation
        file = create_file(path, name + "_myFields", "fld", new_version=new_version)
        dt["myFields"] = file + ".fld"
        import json

        try:
            with open(file + ".fld", "w") as f:
                jsdata = json.dumps(myFields, indent=2)
                f.write(jsdata)
                f.close()
        except:
            raise FileWriteError("Problem saving file " + file + ".fld")

    return dt


# *******************************
@beartype
def export_files(
    name: Optional[str] = "ML",  # Name of Issue
    path: Optional[Union[str, pathlib.PosixPath]] = None,
    validations: Optional[pd.DataFrame] = None,
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    sklearnMl: Optional[Any] = None,
    sklearnOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False,
) -> dict:

    """
        Writes files on disc, like the results of validation, fitted model or OneHotEncoder object.
    Args:
        - name (string, default 'ML'): Name of the ML-process (subject) will be used to build the filenames.
        - path (string, defalut './data'): Name of the output path. If the path does not exists, the function will raise an exception.
        At least one of the following arguments should be not None:
        - validation (DataFrame, default = None): Values of metrics according to the type of the model (classification or regression)
        - crossvalidation (dictionary, default = None): Content of the output file are validation values of each fold.
        - confusion_matrix (DataFrame,  default = None): Content of the output file is a table .
            Exist only for classifier estimators,
            (will be writte in a file just if validation is used).
        - importance (DataFrame,  default = None): Content of the output file is a table with values of importance for each feature.
        - comparison (DataFrame,  default = None): List of all compared value pares: given, predicted
        - sklearnMl (Model Object, default None): Content of the output file, type of the file: SKLAERN Model
        - sklearnOhe (OneHotEncoder Object, default None): Content of the output file, type of the file: SKLEARN OneHotEncoder
        - myFields (dictionary, default = None):  Content of the output file: columnslist with name and type of columns
        - decimalpoint_german (boolen, default False): True if the files above should get "," as decimal point and ";" in csv- files
        - new_version (boolen, default = False): If the file exists it schould be deletet (new_version = False) or a new version shold be created (new_version = True)
    Returns:
        dictionary of the file names of sklearnMl, myOhe, myFields, myMetadata, ...
    """
    # Argument evaluation
    t = (
        sklearnMl.__class__.__name__
    )
    if not (t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression") or sklearnMl is None):
        raise InvalidParameterValueException(
            "Argument sklearnMl ist not in instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    t = sklearnOhe.__class__.__name__
    if not (t in ("OneHotEncoder") or sklearnOhe is None):
        raise InvalidParameterValueException("Argument myOhe ist not in instance of one of OneHotEncoder")

    return _export_files(
        name=name,
        path=path,
        validations=validations,
        crossvalidation=crossvalidation,
        confusion_matrix=confusion_matrix,
        comparison=comparison,
        importance=importance,
        sklearnMl=sklearnMl,
        sklearnOhe=sklearnOhe,
        myFields=myFields,
        decimalpoint_german=decimalpoint_german,
        new_version=new_version,
    )
