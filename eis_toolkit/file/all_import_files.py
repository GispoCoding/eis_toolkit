
from typing import Optional
from pathlib import Path
#import csv
import keras.models as models
from os.path import exists
import json
#import pickle
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _all_import_files(
    sklearnMl_file: Optional[str] = None,
    myOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    #myMetadata_file: Optional[str] = None,
    kerasMl_file: Optional[str] = None, 
    kerasOhe_file: Optional[str] = None
    # more
    # nanmask: Optional[pd.DataFrame] = None 
):
    # Argument evaluation
    fl = []
    if not (isinstance(sklearnMl_file,str) or (sklearnMl_file is None)):          #.__str__()
        fl.append('argument sklearnMl_file is not string and is not None')
    if not (isinstance(myOhe_file,str) or (myOhe_file is None)):
        fl.append('argument myohe_file is not string and is not None')
    if not (isinstance(myFields_file,str) or (myFields_file is None)):
        fl.append('argument myFields_file is not string and is not None')
    # if not (isinstance(myMetadata_file,str) or (myMetadata_file is None)):
    #     fl.append('argument myMetadata_file is not string and is not None')
    if not (isinstance(kerasMl_file,str) or (kerasMl_file is None)):
        fl.append('argument kerasMl_file is not string and is not None')
    if not (isinstance(kerasOhe_file,str) or (kerasOhe_file is None)):
        fl.append('argument kerasOhe_file is not string and is not None')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_import_files: ' + fl[0])
    
    fl = []
    if sklearnMl_file is not None:
        if not exists(sklearnMl_file):
            fl.append('sklearnMl_file does not exists')
    if myOhe_file is not None:
        if not exists(myOhe_file):
            fl.append('myOhe_file does not exists')
    if myFields_file is not None:
        if not exists(myFields_file):
            fl.append('myFieldsnMl_file does not exists')
    # if myMetadata_file is not None:
    #     if not exists(myMetadata_file):
    #         fl.append('myMetadata_file does not exists')
    if kerasMl_file is not None:
        if not exists(kerasMl_file):
            fl.append('skerasMl_file does not exists')
    if kerasOhe_file is not None: 
        if not exists(kerasOhe_file):
            fl.append('kerasOhe_file does not exists')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_import_files: ' + fl[0])

    # Main
    sklearnMl = myOhe = myFields = myMetadata = kerasMl = kerasOhe = None
    # just the last Argument will 
    if sklearnMl_file is not None:               # Model
        sklearnMl = joblib.load(sklearnMl_file)     # load and reuse the model

    if myOhe_file is not None:            # OneHotEncoder
        myOhe = joblib.load(myOhe_file)

    # fields
    if myFields_file is not None:    
        myFields = json.load(open(myFields_file))

    # # metadata
    # if myMetadata_file is not None:       # Validation
    #     myMetadata = joblib.load(myMetadata_file)

    # keras model
    if kerasMl_file is not None:               # Keras Model
        kerasMl = models.load_model(kerasMl_file)     # load and reuse the model

    # ohe for keras classification
    if kerasOhe_file is not None:            # OneHotEncoder for keras classification (y)
        kerasOhe = joblib.load(kerasOhe_file)

    return sklearnMl,myOhe,myFields,kerasMl,kerasOhe

# *******************************
def all_import_files( 
    sklearnMl_file: Optional[str] = None, 
    myOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    #myMetadata_file: Optional[str] = None,
    kerasMl_file: Optional[str] = None, 
    kerasOhe_file: Optional[str] = None
    # ..
):

    """ 
        imports csv and json files as well as objects like model on disc 
    Args:
        - sklearnMl_file (file name, default None): Name of the file of a saved sklearn-model (random forrest or logistic regression)
        - myOhe_file (file name, default None): Name of the file of a saved OneHotEncoding-Object
        - myFields_file (file name, default None): Name of the file of a saved fieldlist
        - kerasMl_file (file name, default None): Name of the file of a saved model
        - kerasOhe_file (file name, default None): Name of the file of a saved OneHotEncoding-Objec for y in case of keras muticlass classification
        just one of this Names should be givn. I case there are more: the last on 
    Returns: 
        one or more of: sklearnMl, myOhe, myFields, kerasMl, kerasOhe (=None, if not loaded)
    """

    sklearnMl,myOhe,myFields,kerasMl,kerasOhe = _all_import_files(
        sklearnMl_file = sklearnMl_file, 
        myOhe_file = myOhe_file,
        myFields_file = myFields_file,
        #myMetadata_file = myMetadata_file,
        kerasMl_file = kerasMl_file, 
        kerasOhe_file = kerasOhe_file
    )

    return sklearnMl,myOhe,myFields,kerasMl,kerasOhe
