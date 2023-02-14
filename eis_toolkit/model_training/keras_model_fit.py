"""
create ann regression based on tensorflow and sklearn
Created an Feb 04 2023
@author: torchala 
""" 


### Experementeller Kern von ANN: Alles instellbar
# ist Kern für ANN-classifcation, -regression und -binaer
# Vorraussetzug;  (1-dimensionales tebnsorflow (aus DataFrame))

from typing import Any, Optional
import pandas as pd
import tensorflow as tf   

from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _keras_model_fit(
    kerasML,
    Xdf: pd.DataFrame,           
    ydf: pd.DataFrame,
    batch_size: Optional[int] = None,   #32, 
    epochs: Optional[int] = 1,              
    verbose: Optional[int] = 'auto'
    #callbacks: Optional[list] = None,
    #validation_split: Optional[float] == 0.0,
    #validation_data: Optional[pd.DataFrame] = None,
    #shuffle: Optional[bool] = True,
    #class_weight: Optional[dict] =None,
    #sample_weight=None,
    #initial_epoch=0,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    #validation_freq=1,
    #max_queue_size=10,
    #workers=1,
    #use_multiprocessing=False,
):

    #  zu tensoren
    x_tens = tf.convert_to_tensor(Xdf)          # schneller ggf.:   tf.constant...
    y_tens = tf.convert_to_tensor(ydf)          #    ""
 
    # columns of y. For prediction-output 

    columns = ydf.columns.tolist()
    # fitten
    history = kerasML.fit(x_tens,y_tens,batch_size=batch_size,epochs=epochs,verbose=verbose)

    return kerasML,history,columns

# *******************************
def keras_model_fit(
    kerasML,
    Xdf: pd.DataFrame,           
    ydf: pd.DataFrame,
    batch_size: Optional[int] = 32, 
    epochs: Optional[int] = 1,              
    verbose: Optional[int] = 'auto'
):
    """ 
        training  a Artifical neuronal network model with Keras

    Args:
        kerasML
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        epochs (int):
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,

    
x: Input data. It could be:
A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights).
A tf.keras.utils.experimental.DatasetCreator, which wraps a callable that takes a single argument of type tf.distribute.InputContext, and returns a tf.data.Dataset. DatasetCreator should be used when users prefer to specify the per-replica batching and sharding logic for the Dataset. See tf.keras.utils.experimental.DatasetCreator doc for more information. A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is given below. If these include sample_weights as a third component, note that sample weighting applies to the weighted_metrics argument but not the metrics argument in compile(). If using tf.distribute.experimental.ParameterServerStrategy, only DatasetCreator type is supported for x.
y: Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided (unless the steps_per_epoch flag is set to something other than None). Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).
callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See tf.keras.callbacks. Note tf.keras.callbacks.ProgbarLogger and tf.keras.callbacks.History callbacks are created automatically and need not be passed into model.fit. tf.keras.callbacks.ProgbarLogger is created or not based on verbose argument to model.fit. Callbacks with batch-level calls are currently unsupported with tf.distribute.experimental.ParameterServerStrategy, and users are advised to implement epoch-level calls instead with an appropriate steps_per_epoch value.
validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a dataset, generator or keras.utils.Sequence instance. If both validation_data and validation_split are provided, validation_data will override validation_split. validation_split is not yet supported with tf.distribute.experimental.ParameterServerStrategy.
validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. Thus, note the fact that the validation loss of data provided using validation_split or validation_data is not affected by regularization layers like noise and dropout. validation_data will override validation_split. validation_data could be: - A tuple (x_val, y_val) of Numpy arrays or tensors. - A tuple (x_val, y_val, val_sample_weights) of NumPy arrays. - A tf.data.Dataset. - A Python generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights). validation_data is not yet supported with tf.distribute.experimental.ParameterServerStrategy.
shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). This argument is ignored when x is a generator or an object of tf.data.Dataset. 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
sample_weight: Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. This argument is not supported when x is a dataset, generator, or keras.utils.Sequence instance, instead provide the sample_weights as the third element of x. Note that sample weighting does not apply to metrics specified via the metrics argument in compile(). To apply sample weighting to your metrics, you can specify them via the weighted_metrics in compile() instead.
initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
steps_per_epoch: Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input dataset is exhausted. When passing an infinitely repeating dataset, you must specify the steps_per_epoch argument. If steps_per_epoch=-1 the training will run indefinitely with an infinitely repeating dataset. This argument is not supported with array inputs. When using tf.distribute.experimental.ParameterServerStrategy: * steps_per_epoch=None is not supported.
validation_steps: Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted. In the case of an infinitely repeated dataset, it will run into an infinite loop. If 'validation_steps' is specified and only part of the dataset will be consumed, the evaluation will start from the beginning of the dataset at each epoch. This ensures that the same validation samples are used every time.
validation_batch_size: Integer or None. Number of samples per validation batch. If unspecified, will default to batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
validation_freq: Only relevant if validation data is provided. Integer or collections.abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
max_queue_size: Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
workers: Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
use_multiprocessing: Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.

    Returns:
        model, history, columns
    """
    kerasML,history,columns = _keras_model_fit(
        kerasML = kerasML,
        Xdf = Xdf,                      # ggf. in model_fit auslagerbn
        ydf = ydf,
        batch_size = batch_size,
        epochs = epochs,                # max_iter       (bei sklearn in de funktion bei tensorflow bei model.fit)
        verbose = verbose
    )

    return kerasML,history,columns

