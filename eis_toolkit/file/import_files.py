
from typing import Optional
import keras.models as models
from os.path import exists
import json
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException, MissingFileOrPath, FileReadWriteError

# *******************************
def _import_files(
    sklearnMl_file: Optional[str] = None,
    sklearnOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    kerasMl_file: Optional[str] = None, 
    kerasOhe_file: Optional[str] = None,
    # nanmask: Optional[pd.DataFrame] = None 
):
    
    # Main
    sklearnMl = sklearnOhe = myFields = kerasMl = kerasOhe = None
    # just the last Argument will 
    if sklearnMl_file is not None:               # Model
        try:
            sklearnMl = joblib.load(sklearnMl_file)     # load and reuse the model
        except: 
            raise FileReadWriteError('sklearnMl file is not readable '+ str(sklearnMl_file))
    if sklearnOhe_file is not None:            # OneHotEncoder
        try:
            sklearnOhe = joblib.load(sklearnOhe_file)
        except:
            raise FileReadWriteError('sklearnOhe file is not readable '+ str(sklearnOhe_file))

    # fields
    if myFields_file is not None:
        try:
            myFields = json.load(open(myFields_file))
        except:
            raise FileReadWriteError('fields file is not readable '+str(myFields_file))

    # keras model
    if kerasMl_file is not None:               # Keras Model
        try:
            kerasMl = models.load_model(kerasMl_file)     # load and reuse the model
        except:
            raise FileReadWriteError('kerasMl file is not readable '+str(kerasMl_file))

    # ohe for keras classification
    if kerasOhe_file is not None:            # OneHotEncoder for keras classification (y)
        try:
            kerasOhe = joblib.load(kerasOhe_file)
        except:
            raise FileReadWriteError('kerasOhe file is not readable'+str(kerasOhe_file))

    return sklearnMl, sklearnOhe, myFields, kerasMl, kerasOhe

# *******************************
def import_files( 
    sklearnMl_file: Optional[str] = None, 
    sklearnOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    kerasMl_file: Optional[str] = None, 
    kerasOhe_file: Optional[str] = None
    # ..
):

    """ 
        imports csv and json files as well as objects like model on disc 
    Args:
        - sklearnMl_file (file name, default None): Name of the file of a saved sklearn-model (random forrest or logistic regression)
        - sklearnOhe_file (file name, default None): Name of the file of a saved OneHotEncoding-Object
        - myFields_file (file name, default None): Name of the file of a saved fieldlist (Dictionary)
        - kerasMl_file (file name, default None): Name of the file of a saved model
        - kerasOhe_file (file name, default None): Name of the file of a saved OneHotEncoding-Objec for y in case of keras muticlass classification
        just one of this Names should be givn. I case there are more: the last on 
    Returns: 
        one or more of (=None, if not loaded))
        - SKLEARN Model, 
        - SKLEARN OneHotEncoder,
        - Dictionary of Fields, 
        - KERAS Model, 
        - KERAS OneHotEncoder
    """

    # Argument evaluation
    fl = []
    if not (isinstance(sklearnMl_file, str) or (sklearnMl_file is None)):
        fl.append('Argument sklearnMl_file is not string and is not None')
    if not (isinstance(sklearnOhe_file, str) or (sklearnOhe_file is None)):
        fl.append('Argument sklearnohe_file is not string and is not None')
    if not (isinstance(myFields_file, str) or (myFields_file is None)):
        fl.append('Argument myFields_file is not string and is not None')
    if not (isinstance(kerasMl_file, str) or (kerasMl_file is None)):
        fl.append('Argument kerasMl_file is not string and is not None')
    if not (isinstance(kerasOhe_file, str) or (kerasOhe_file is None)):
        fl.append('Argument kerasOhe_file is not string and is not None')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function import_files: ' + fl[0])
    
    fl = []
    if sklearnMl_file is not None:
        if not exists(sklearnMl_file):
            fl.append('sklearnMl_file does not exists')
    if sklearnOhe_file is not None:
        if not exists(sklearnOhe_file):
            fl.append('sklearnOhe_file does not exists')
    if myFields_file is not None:
        if not exists(myFields_file):
            fl.append('myFieldsnMl_file does not exists')
    if kerasMl_file is not None:
        if not exists(kerasMl_file):
            fl.append('skerasMl_file does not exists')
    if kerasOhe_file is not None: 
        if not exists(kerasOhe_file):
            fl.append('kerasOhe_file does not exists')
    if len(fl) > 0:
        raise MissingFileOrPath(fl[0])

    sklearnMl, sklearnOhe, myFields, kerasMl, kerasOhe = _import_files(
        sklearnMl_file = sklearnMl_file, 
        sklearnOhe_file = sklearnOhe_file,
        myFields_file = myFields_file,
        kerasMl_file = kerasMl_file, 
        kerasOhe_file = kerasOhe_file
    )

    return sklearnMl, sklearnOhe, myFields, kerasMl, kerasOhe
