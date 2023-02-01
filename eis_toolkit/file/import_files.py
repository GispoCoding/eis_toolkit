
from typing import Optional, Any
from pathlib import Path
import csv
#import pandas
import os
import json
#import pickle
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _import_files(
    myML_file: Optional[str] = None,
    myOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    myMetadata_file: Optional[str] = None
    # more
    # nanmask: Optional[pd.DataFrame] = None 
):
    myML = myOhe = myFields = myMetadata = None
    # just the last Argument will 
    if myML_file is not None:               # Model
        myML = joblib.load(myML_file)     # load and reuse the model

    if myOhe_file is not None:            # OneHotEncoder
        myOhe = joblib.load(myOhe_file)

    # fields
    if myFields_file is not None:    
        myFields = json.load(open(myFields_file))

    # metadata
    if myMetadata_file is not None:       # Validation
        myMetadata = json.load(open(myMetadata_file))

    return myML,myOhe,myFields,myMetadata

# *******************************
def import_files( 
    myML_file: Optional[str] = None, 
    myOhe_file: Optional[str] = None,
    myFields_file: Optional[str] = None,
    myMetadata_file: Optional[str] = None
    # ..
):

    """ 
        imports csv and json files as well as objects like model on disc 
    Args:
        myML_name (file name, default None): Name of the file of a saved model
        myOhe_name (file name, default None): Name of the file of a saved OneHotEncoding-Object
        myFields_name (file name, default None): Name of the file of a saved fieldlist
        myMetadata_name (file name, default None): Name of the file of saved Metadata
        just one of this Names should be givn. I case there are more: the last on 
    Returns: myData  (one of: myML, myOhe, myFields, myMetadata)
    """

    myML,myOhe,myFields,myMetadata = _import_files(
        myML_file = myML_file, 
        myOhe_file = myOhe_file,
        myFields_file = myFields_file,
        myMetadata_file = myMetadata_file
    )

    return myML,myOhe,myFields,myMetadata
