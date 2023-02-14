"""
prediction for a keras-model
Created an Feb 04 2023
@author: torchala 
""" 

from typing import Any, Optional, Tuple
import pandas as pd
import tensorflow as tf
import numpy as np
# from keras.models import Sequential
# from keras import predict      

from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _keras_model_predict(
    kerasML,
    Xdf: pd.DataFrame,
    igdf: Optional[int] = None,
    columns: Optional[int] = None,    # columns names of y, the predicted columns 
    ohe: Optional[int] = None,
    batch_size: Optional[int] = None, 
    verbose: Optional[int] = 'auto',
    steps: Optional[int] = None 
) -> Tuple[pd.DataFrame,pd.DataFrame]: 

    #  zu tensoren
    x_tens = tf.convert_to_tensor(Xdf)
 
    # fitten
    ypr = kerasML.predict(x_tens,batch_size=batch_size,verbose=verbose,steps=steps)

    # output as Dataframe and zipped with 
    # if columns is None:
    #     ypr = pd.DataFrame(ypr)
    #     cl = ypr.columns  #.tolist()
    # else:
    if ohe is not None:
        cl = ohe.categories_[0].tolist()
        ypr = pd.DataFrame(ypr,columns=cl)
    else: 
        ypr = pd.DataFrame(ypr)
    # result-column: colmns name of max value
    if ypr.shape[1] > 1:
        ypr['result'] = ypr.apply('idxmax', axis=1)
        ydf = ypr['result']
        if igdf is not None and ypr is not None:
            if len(pd.DataFrame(ypr).index) != len(igdf.index):
                raise InvalidParameterValueException ('***  ydf and igdf have different number of rows')
            elif len(igdf.columns) > 0:
                ypr =  pd.DataFrame(np.column_stack((igdf,ypr)),columns=igdf.columns.to_list()+ypr.columns.to_list())
        return ydf,ypr
    else:
        return ypr, None

    # result-columns
    # if ypr.shape[1] > 1:
    #     ydf = ypr.max(axis = 1)
    #     return ydf,ypr
    # return ypr, None

# *******************************
def keras_model_predict(
    kerasML,
    Xdf: pd.DataFrame,
    igdf: Optional[int] = None,
    columns: Optional[int] = None,
    ohe: Optional[int] = None,
    batch_size: Optional[int] = None,           
    verbose: Optional[int] = 'auto',
    steps: Optional[int] = None
    # callback = Optional[int] = None
) -> Tuple[pd.DataFrame,pd.DataFrame]: 

    """ 
        training  a Artifical neuronal network model with Keras

    Args:
        kerasMl
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        igdf (Pandas dataframe):  Identification and geoemtry-columns 
        batch_size=None,
        verbose="auto",
        steps=None,
    # callbacks=None,
    # max_queue_size=10,
    # workers=1,
    # use_multiprocessing=False,

A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
A tf.data dataset.
A generator or keras.utils.Sequence instance. A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is given in the Unpacking behavior for iterator-like inputs section of Model.fit.
batch_size: Integer or None. Number of samples per batch. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of dataset, generators, or keras.utils.Sequence instances (since they generate batches).
verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. "auto" defaults to 1 for most cases, and to 2 when used with ParameterServerStrategy. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (e.g. in a production environment).
steps: Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None. If x is a tf.data dataset and steps is None, predict() will run until the input dataset is exhausted.
callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during prediction. See callbacks.

    
    Returns:
        result(y), prediction
    """
    ydf,ypr = _keras_model_predict(
        kerasML = kerasML,
        Xdf = Xdf,
        igdf = igdf,
        columns = columns,
        ohe = ohe,
        batch_size = batch_size,
        verbose = verbose,
        steps = steps
    )
    return ydf, ypr

