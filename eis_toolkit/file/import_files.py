from beartype import beartype
from beartype.typing import Optional, Union
import pathlib
from os.path import exists
import json
import joblib
from eis_toolkit.exceptions import FileReadError


# *******************************
@beartype
def _import_files(
    sklearnMl_file: Optional[Union[str, pathlib.PosixPath]] = None,
    sklearnOhe_file: Optional[Union[str, pathlib.PosixPath]] = None,
    myFields_file: Optional[Union[str, pathlib.PosixPath]] = None,
):

    # Main
    sklearnMl = sklearnOhe = myFields = None
    # just the last Argument will
    if sklearnMl_file is not None:               # Model
        try:
            sklearnMl = joblib.load(sklearnMl_file)     # load and reuse the model
        except:  # noqa: E722
            raise FileReadError('sklearnMl file is not readable ' + str(sklearnMl_file))
    if sklearnOhe_file is not None:            # OneHotEncoder
        try:
            sklearnOhe = joblib.load(sklearnOhe_file)
        except:  # noqa: E722
            raise FileReadError('sklearnOhe file is not readable ' + str(sklearnOhe_file))

    # fields
    if myFields_file is not None:
        try:
            myFields = json.load(open(myFields_file))
        except:  # noqa: E722
            raise FileReadError('fields file is not readable ' + str(myFields_file))

    return sklearnMl, sklearnOhe, myFields,


# *******************************
@beartype
def import_files(
    sklearnMl_file: Optional[Union[str, pathlib.PosixPath]] = None,
    sklearnOhe_file: Optional[Union[str, pathlib.PosixPath]] = None,
    myFields_file: Optional[Union[str, pathlib.PosixPath]] = None,
):
    """
        Read files to objects like randmom forest model.

        These files are creted during training of the model.

    Args:
         sklearnMl_file (file name, default None):
            Name of the file of a saved sklearn-model (random forrest or logistic regression)
        - sklearnOhe_file (file name, default None): Name of the file of a saved OneHotEncoding-Object
        - myFields_file (file name, default None): Name of the file of a saved fieldlist (Dictionary)

    Returns:
        one or more of (=None, if not loaded))
        - SKLEARN Model,
        - SKLEARN OneHotEncoder,
        - Dictionary of Fields,
    """

    # Argument evaluation
    if sklearnMl_file is not None:
        if not exists(sklearnMl_file):
            raise FileReadError('sklearnMl_file does not exists')
    if sklearnOhe_file is not None:
        if not exists(sklearnOhe_file):
            raise FileReadError('sklearnOhe_file does not exists')
    if myFields_file is not None:
        if not exists(myFields_file):
            raise FileReadError('myFieldsnMl_file does not exists')

    sklearnMl, sklearnOhe, myFields = _import_files(
        sklearnMl_file=sklearnMl_file,
        sklearnOhe_file=sklearnOhe_file,
        myFields_file=myFields_file,
    )

    return sklearnMl, sklearnOhe, myFields,
