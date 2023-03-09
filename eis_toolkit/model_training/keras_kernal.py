
from typing import Optional
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import keras.optimizers as optimizers #import Adam
import keras.losses as losses
import keras.activations as activations
import keras.metrics as metrics

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
def _keras_kernal(
    Xdf: Optional[pd.DataFrame] = None,             # getting shape for the inputlayer (number of features)
    ydf: Optional[pd.DataFrame] = None,             # getting number of classes for last layer-unit 
    # input (first layer)
    input_dim: Optional[list] = None,    # number of features,  Xdf.shape[1] = shape (one if both should be given)
    # dtype: Optional[str] = None, 	# e.g. float32
    # sparse: Optional[bool] = None,  # default: Flase
    # shape: Optional[int] = None,             # if None, Xdf should be given
    # activation1: Optional[int] = None,   
    # learning_rate: float,       # learning_rate_init 
    # layers
    layer_nodes: Optional[dict] = None,           # hidden_layer_sizes (units), activation, use_bias 
                                # activation:  relu f. all hiddenlyer, 
                                #  outut-Layer: 'linear' for regression, 'sigmoid' for binary, 'softmax' for classification
    #last_layer:
    units_output: Optional[int] = None,
    activation_output: Optional[str] = None,
    use_bias_output: Optional[bool] = True,
    #compile
    optimizer: Optional[str] = 'adam', # "rmsprop" (default),  oder 'sgd' als abschluss
    learning_rate: Optional[str] = None,    # default: 0.001
    loss: Optional[str] = 'categoricalcrossentropy',      #=None,  'categorical_crossentropy'  für regression: z. B. 'mean_squared_error'
    metric: Optional[list] = ['accuracy']   # =None,  'mean_squared_error' ('mse') 'CategoricalAccuracy'
    #loss_weights: Optional[list] =None, 
):

    if layer_nodes == None:
        raise InvalidParameterValueException ('***  layer_nodes is empty ')

    if optimizer is None:
        optimizer = 'adam'    #optimizer='SGD' or 'adam' #(learning_rate=learning_rate) 
    if optimizer.lower() == 'adam':
        opt=optimizers.Adam(learning_rate=learning_rate) #, epsilon, ema_momentum, .... 
    elif optimizer.lower() == 'sgd':
        opt=optimizers.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'adadelta':
        opt=optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer.lower() == 'adagrad':
        opt=optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer.lower() == 'adamax':
        opt=optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer.lower() == 'ftrl':
        opt=optimizers.Ftrl(learning_rate=learning_rate)
    elif optimizer.lower() == 'nadam':
        opt=optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt=optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise InvalidParameterValueException ('***  unknown optimizer')
    
    if loss is None:
        loss = 'meansquarederror'
    if loss.lower() in ('meansquarederror','mean_squared_error','mse'):                  # for regression (for multiclass classification ist possible too)
        lss = losses.MeanSquaredError()
    elif loss.lower() in ('categorical_crossentropy','categoricalcrossentropy'):          # for classification (multiclass, binary and multilabel)
        lss = losses.CategoricalCrossentropy()
    elif loss.lower() in ('mean_absolute_error','meanabsoluteerror','mae'):               # for regression
        lss = losses.MeanAbsoluteError()
    elif loss.lower() in ('mean_squared_logarithmic_error','meansquaredlogarithmicerror','msle'):
        lss = losses.MeanSquaredLogarithmicError()
    elif loss.lower() in ('sparse_categorical_crossentropy','sparsecategoricalcrossentropy'):
        lss = losses.SparseCategoricalCrossentropy()
    elif loss.lower() in ('binary_crossentropy','binarycrossentropy'):                  # for binary classification (Arguments: e.g. from_logits)
        lss = losses.BinaryCrossentropy()    
    # BinaryCrossentropy, BineryFocalCrossentropy, CetegoricalHinge, Hinge, SquerdHinge, Huber, KLDivergenc, Poisson, 
    else:
        raise InvalidParameterValueException ('***  unknown loss')

    # activation for MLP:
    # elu  exponential Linar Unit      Arguments: alpha (float)
    # exponential
    # gelu  gaussian error linear unit
    # hard_sigmoid
    # linear                                    for regression
    # relu   rectified linear unit              for hidden layer     Arguments: alpha (float), max_value (float), threshold (float)
    # selu   scaled exponential linear unit
    # sigmoid                                 for output layer
    #                                         for binary classification (buy or not to buy) -> probabilie
    #                                         for multilabel classification (propabilit nor the one yes/no), e.g. HarraPotter is advanture: yes, childern: yes, clime: no....
    # softmax   probability destribution      for output layer      arguments: axis (integer)
    #                                         for multiclass classification (propability:  cat?, dog?,...) sum of propabilies is =1
    # softplus
    # softsign
    # swish                                     (for hidden layer)
    # advanced activatioN. e.g. ERelu, LeakyReLU

    if activation_output == None:
        activation_output = 'softmax'
    if activation_output.lower()  in ('softmax'):
        al = activations.softmax
    elif activation_output.lower()  in ('sigmoid'):
        al = activations.sigmoid
    elif activation_output.lower()  in ('linear'):
        al = activations.linear

    # metrics
    if metric is None:
        metric = ['accuracy']
    elif len(metric) == 0:
        metric = ['accuracy']
    metr = []
    for m in metric:    
        # regression R2, mse, rmse, mae, me, mape                   
        # classification;  Accuracy. 
        # Confusion Matrix. ...
        # AUC/ROC. ...
        # Precision. ...
        # Recall. ...
        # F1 score. ...              binary classification
        # Kappa.
        # 
        if m.lower() in ('accuracy'):                # for classification  (F1: a funktion from Recall and Precision sould be defined)
            metr.append(metrics.Accuracy())
        elif m.lower()  in ('binaryaccuracy','binary_accuracy'):
            metr.append(metrics.BinryAccuracy())
        elif m.lower()  in ('binarycrossentropy','binary_crossentropy'):       # arguments: from_logists (False), label_smoothing (0)
            metr.append(metrics.BinaryCrossentropy())
        elif m.lower()  in ('auc'):       # for binray classificatio n  arguments: from_logists (False), label_smoothing (0),........
            metr.append(metrics.AUC())
        elif m.lower()  in ('mse','meansquarederror','mean_squared_error'):     # for regression
            metr.append(metrics.MeanSquaredError())
        elif m.lower() in ('categoricalaccuracy','categorical_accuracy'):               
            metr.append(metrics.CategoricalAccuracy())
        elif m.lower() in ('categoricalcrossentropy','categorical_crossentropy'):         # for multilabel classification  (F1: a funktion from Recall and Precision sould be defined)
            metr.append(metrics.CategoricalCrossentropy())            
        # ... a lot more
    if metr is None:
        raise InvalidParameterValueException ('***  unknown metrics')
    # input_dim
    if input_dim is None:
        if Xdf is None:
            raise InvalidParameterValueException ('***  shape and Xdf of input are None')
        #x_tens = tf.convert_to_tensor(Xdf)
        input_dim = Xdf.shape[1]
    if units_output is None:
        if ydf is None:
            raise InvalidParameterValueException ('***  input and ydf are None')
        #y_tens = tf.convert_to_tensor(ydf)   
        #units_output = ydf.apply(pd.value_counts).shape[1]  # number of classes (on column)
        units_output = ydf.shape[1]   # number of columns
    # Model Placeholder
    kerasML = Sequential()

    #myML.add(Input(shape = (shape,),dtype = dtype, sparse = sparse ))                # batch_size=batch_size, Eingabe-Tensor   auch  input_dim=xl   
    q = True                                            #  oder als erster Dense-Layer in der Schleife: myML.add(Dense(units = i, input_dim=xl, activation= 'relu')) 
    for i in layer_nodes:
        if q:    # first layer
            kerasML.add(Dense(input_dim=input_dim,units=i['units'],activation=i['activation'], use_bias=i['use_bias'])) 
            q = False
        else:
            kerasML.add(Dense(units=i['units'],activation=i['activation'], use_bias=i['use_bias']))
            # in case we want to use obects: the elif-cases should be codd in a function which sulkd be callesd here. 
            # the result act should be coded in activation-Argument: acitication = act
    #outpu-layer
    kerasML.add(Dense(units=units_output,activation=al,use_bias=use_bias_output))
    
    #Metrix
    kerasML.compile(loss=lss,   #'categorical_crossentropy', #tf.losses.mean_squared_error,    #MeanSquaredError()
                                                # 'mean_absolute_error', 'mean_squared_logarithmic_error', 
                 optimizer=opt,              #optimizer='SGD' or 'adam' #(learning_rate=learning_rate)  #,metrics=[r2_metric])
                 metrics=metr
    )

    return kerasML

# *******************************
def keras_kernal(
    Xdf: Optional[pd.DataFrame] = None,
    ydf: pd.DataFrame = None,
    # input
    input_dim: Optional[list] = None,    # Xdf.shape[1]  should 
    # batch_size: Optional[int] = 32, # Optional 32
    # dtype: Optional[str] = None, 	# e.g. float32
    # sparse: Optional[bool] = None,  # default: Flase
    # learning_rate: float,       # learning_rate_init 
    # layers
    layer_nodes: Optional[dict] = None,           # hidden_layer_sizes (units), activation, use_bias 
                                # activation:  relu f. all hiddenlyer, 
                                #  outut-Layer: 'linear' for regression, 'sigmoid' for binary, 'softmax' for classification
    #last_layer:
    units_output: Optional[int] = None,
    activation_output: Optional[str] = None,
    use_bias_output: Optional[bool] = True,
    #compile
    optimizer: Optional[str] = None, # 'adam, "rmsprop",  oder 'sgd' als abschluss
    learning_rate: Optional[str] = None,
    loss: Optional[str] = None,      #=None,  'categorical_crossentropy'  für regression: z. B. 'mean_squared_error'
    metric: Optional[list] = None #['accuracy']   # =None,  'mean_squared_error' ('mse')
    #loss_weights: Optional[list] =None, 
):


    """ 
        creating a Artifical neuronal network model with Keras
    Args 
        Xdf to get input-shape (number of features)
        ydf to get number of nods in last Danse-Layer (number of classes), just for classification
        - Parameters of an InputLayer (default = None)
            - input_shape	Shape tuple (not including the batch axis), or TensorShape instance (not including the batch axis).
            - batch_size	Optional input batch size (integer or None).
            - dtype	Optional datatype of the input. When not provided, the Keras default float type will be used.
            # - input_tensor	Optional tensor to use as layer input. If set, the layer will use the tf.TypeSpec of this tensor rather than creating a new placeholder tensor.
            - sparse	Boolean, whether the placeholder created is meant to be sparse. Default to False.
            # - ragged	Boolean, whether the placeholder created is meant to be ragged. In this case, values of None in the shape argument represent ragged dimensions. For more information about tf.RaggedTensor, see this guide. Default to False.
            # - type_spec	A tf.TypeSpec object to create Input from. This tf.TypeSpec represents the entire batch. When provided, all other args except name must be None.
            # - name	Optional name of the layer (string).
        - add dense layer: (default = None)
        - leyer_nodes: Dictionary with folloing keys:
                - units	(no default) Positive integer, dimensionality of the output space.
                - activation	Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
                - use_bias	Boolean, whether the layer uses a bias vector.
            # - kernel_initializer (default = 'glorot_uniform')	Initializer for the kernel weights matrix.
            # - bias_initializer (default = 'zeros')	Initializer for the bias vector.
            # - kernel_regularizer	Regularizer function applied to the kernel weights matrix.
            # - bias_regularizer	Regularizer function applied to the bias vector.
            # - activity_regularizer	Regularizer function applied to the output of the layer (its "activation").
            # - kernel_constraint	Constraint function applied to the kernel weights matrix.
            # - bias_constraint	Constraint function applied to the bias vector.
        - last_layer:
            - units	(no default) Positive integer, dimensionality of the output space. If None ydf should be given
                             For regression: always =1
            - activation	Activation function to use. If not specify, no activation is applied (ie. "linear" activation: a(x) = x).
            - use_bias	Boolean, whether the layer uses a bias vector.
        - compile:
            - optimizer: String ( default = 'Adams', name of optimizer) or optimizer instance. See tf.keras.optimizers.
            - loss: Loss function. May be a string (name of loss function), or a tf.keras.losses.Loss instance. See tf.keras.losses. A loss function is any callable with the signature loss = fn(y_true, y_pred), where y_true are the ground truth values, and y_pred are the model's predictions. y_true should have shape (batch_size, d0, .. dN) (except in the case of sparse loss functions such as sparse categorical crossentropy which expects integer arrays of shape (batch_size, d0, .. dN-1)). y_pred should have shape (batch_size, d0, .. dN). The loss function should return a float tensor. If a custom Loss instance is used and reduction is set to None, return value has shape (batch_size, d0, .. dN-1) i.e. per-sample or per-timestep loss values; otherwise, it is a scalar. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses, unless loss_weights is specified.
            - metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy based on the shapes of the targets and of the model output. We do a similar conversion for the strings 'crossentropy' and 'ce' as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the weighted_metrics argument instead.
            - loss_weights: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected to map output names (strings) to scalar coefficients.
            - weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
    Returns:
        model
    """

    kerasML = _keras_kernal(
        Xdf = Xdf,
        ydf = ydf,
        input_dim = input_dim,
        layer_nodes = layer_nodes,
        units_output = units_output,
        activation_output = activation_output,
        use_bias_output = use_bias_output,
        optimizer = optimizer,
        learning_rate = learning_rate,
        loss = loss,
        metric = metric
    )

    return kerasML

