
#### not tested version
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input

#from tensorflow import GradientTape           

from eis_toolkit.exceptions import InvalidParameterValueException

    # hidden_layer_sizes: Optional[tuple] =(100,),
    # activation: Optional[activation] = "relu",
    # solver: Optional[activation]="adam",
    # alpha: Optional[float] = 0.0001,
    # batch_size: Optional [int | str] ="auto",
    # learning_rate: Optional[learning] = "constant",
    # learning_rate_init: Optional[float] = 0.001,
    # power_t: Optional[float] = 0.5,
    # max_iter: Optional[int] = 200,

# *******************************
def _ann_regressor(
    Xdf: pd.DataFrame,             # ggf. in model_fit auslagerbn
    ydf: pd.DataFrame,
    epochs: int,                # max_iter       (bei sklearn in de funktion bei tensorflow bei model.fit)
    #learning_rate: float,       # learning_rate_init 
    layer_nodes: list,           # hidden_layer_sizes
    verbose: Optional[bool] =False, 
    activations: Optional[list] = None
) -> Any:

    # auf cpu
    # tf.config.set_visible_devices([], 'GPU')
    # Hide GPU from visible devices

    #  zu tensoren
    x_tens = tf.convert_to_tensor(Xdf)
    y_tens = tf.convert_to_tensor(ydf)
 
    # Model Platzhalter 
    myML = Sequential()
    xl = x_tens.shape[1]
    myML.add(Input(shape =(xl,)))   # Eingabe-Tensor   auch  input_dim=xl   
        #  oder als erster Dense-Layer in der Schleife: myML.add(Dense(units = i, input_dim=xl, activation= 'relu')) 
        #q = False
    for i in layer_nodes:
        # if q:
        #     myML.add(Dense(units = i, input_dim=xl, activation= 'relu'))
        #     q = False
        myML.add(Dense(units = i, activation= 'relu'))       #'sigmoid'))
    myML.add(Dense(1, activation="linear"))    # for regression 
    #myML.add(Dense(units = y_tens.shape[1], activation='softmax'))   #'sigmoid')) # Output laye   , bei Classification: Anzahl der Classen

    myML.compile(loss = 'mean_squared_error', #'categorical_crossentropy', #tf.losses.mean_squared_error,    #MeanSquaredError()
                 optimizer='Adam'       #tf.keras.optimizers.Adam) #(learning_rate=learning_rate)  #,metrics=[r2_metric])
    )
    # fitten
    history = myML.fit(Xdf,ydf,epochs = epochs, verbose = verbose)

    # plt

    plt.ylabel('loss')
    plt.ylabel('epochs')
    plt.plot(history.history['loss'])
    plt.show(block=True)

    # Evaluation
    #loss, r2 = myML.evaluation(Xdf,ydf,verbose=verbose)

    return myML

# *******************************
def ann_regressor(
    Xdf: pd.DataFrame,             # ggf. in model_fit auslagerbn
    ydf: pd.DataFrame,
    epochs: int,                # max_iter       (bei sklearn in de funktion bei tensorflow bei model.fit)
    #learning_rate: float,       # learning_rate_init 
    layer_nodes: list,           # hidden_layer_sizes
    verbose: Optional[bool] =False, 
    activations: Optional[list] = False 
) -> Any:

    """ 
    Test of artifitial networks
    Args:
        Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (raws)
        epochs (int):
        lerning rate:
        layer_nodes (list): Anzahl der nodes pro layer

    Returns:
        model 
    """

    myML = _ann_regressor(
        Xdf = Xdf,             # ggf. in model_fit auslagerbn
        ydf = ydf,
        epochs = epochs,                # max_iter       (bei sklearn in de funktion bei tensorflow bei model.fit)
        #learning_rate = learning_rate,       # learning_rate_init 
        layer_nodes = layer_nodes,           # hidden_layer_sizes
        verbose = verbose
    )

    return myML

