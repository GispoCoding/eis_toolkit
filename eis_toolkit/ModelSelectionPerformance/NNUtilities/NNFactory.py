import traceback

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from eis_toolkit.exceptions import CanNotMakeCategoricalLabelException
import numpy as np 
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

class NNFactory():
    """
    This function is used to instance a network
    Methods:
        make_the_one_hot_encoding(self, labels_to_transform)
        generate_dynamically_the_MLP(self, input_channel_1: tuple, rescaling: bool ,dense_nodes: list, final_output: int, last_activation: str, l_2: float):
        do_the_fitting_with_microprocessing(self, worker, configuration: dict, cnn_configuration_dict: dict, X_train, y_train, X_test, y_test,weighted_samples):
    """
    def __init__(self, configuration_handler) -> None:
        """
        Args:
        configuration (dict) a dictionary with all the needed config:
        cnn_configuration (dict) the cnn config :
        """
        self.configuration = configuration_handler.configuration
        self.cnn_configuration = configuration_handler.cnn_parameters_from_ht
        self.instance_of_the_model = None

    def make_the_one_hot_encoding(self, labels_to_transform):
        """
            you need to make one hot encoding because we use the stragety 2 classes and softmax activation
            Args:
                labels_to_transform (np.array): array of label to do the encoding
            Return:
                labels_to_transform (np.array): encoded labels
        """
        enc = OneHotEncoder(handle_unknown='ignore')
        temp = np.reshape(labels_to_transform, (-1, 1))
        labels_to_transform = enc.fit_transform(temp).toarray()

        if (labels_to_transform.sum(axis=1) - np.ones(labels_to_transform.shape[0])).sum() != 0:
            raise CanNotMakeCategoricalLabelException
        
        return labels_to_transform

    def generate_dynamically_the_MLP(self, input_channel_1: tuple, rescaling: bool ,dense_nodes: list,
                                     final_output: int, last_activation: str, l_2: float) -> tf.keras.Model:
        """
        This function construct a MLP
        Args:
            input_channel_1 (np.array):
            rescaling (bool): if you want to normalize the data (1/255)
            dense_nodes (list):  list of neurons
            final_output (int): how many final input you need
            last_activation: (str): softmax | sigmoid
            l_2 (float): L2 regularization value
        Results:
            model (keras model): a compiled model with the provided data
        """

        try:
            input_layer = tf.keras.Input(shape=input_channel_1, name="input_1")
            """
            if rescaling:
                body = tf.keras.layers.Rescaling(1.0 / 255)(input_layer)
                # one conv 1d example
                # we need flatten
                flatten = tf.keras.layers.Flatten(name=f"flatten_layer")(body)
            else:
                flatten = tf.keras.layers.Flatten(name=f"flatten_layer")(input_layer)
            """
            if len(dense_nodes) > 0:
                # add the classifier
                for ctn, dense_unit in enumerate(dense_nodes):
                    if ctn == 0:
                        classifier = tf.keras.layers.Dense(dense_unit,
                                                           activation='relu',
                                                           kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                           bias_regularizer=tf.keras.regularizers.L2(l2=l_2))(input_layer)
                    else:
                        classifier = tf.keras.layers.Dense(dense_unit,
                                                           activation='relu',
                                                           kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                           bias_regularizer=tf.keras.regularizers.L2(l2=l_2))(classifier)

                        # add the final output
                classifier = tf.keras.layers.Dense(final_output,
                                                   activation=last_activation,
                                                   kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   name=f"final_classifier")(classifier)
            else:
                # add the final output
                classifier = tf.keras.layers.Dense(final_output,
                                                   activation=last_activation,
                                                   kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   name=f"final_classifier")(input_layer)

            # create the model obj
            model = tf.keras.Model(inputs=input_layer, outputs=classifier, name=f"model_1")
            print(f"[NN FACTORY] The model is created")
            self.instance_of_the_model = model
            return model
        except Exception as ex:
            print(f"[EXCEPTION] Create dynamically the model throws exception {ex}")

    def do_the_fitting_with_microprocessing(self,
                                            worker,
                                            X_train,
                                            y_train,
                                            X_test,
                                            y_test
                                            ):
        """
        Do the fitting multi-processing fit the model
        Args:
            worker (int): Worker number from multiprocessing
            X_train (np.array): X train array
            y_train (np.array): y train array
            X_test (np.array or None): x test if it is present r none
            y_test (np.array or None): y test if it is present or none
            weighted_samples (bool): if we have to weights samples or not
        Returns:
            list of
            instance of the model (tf.keras.Model)
            scores (float): score of the model
            history accuracy
            true labels (if X test is not None)
            predictions (if y test is not None)
        Raise:
            generic exception
        """

        self.cnn_configuration["input_channel_1"] = (X_train.shape[1])

        # load the model
        self.generate_dynamically_the_MLP(**self.cnn_configuration)

        if self.cnn_configuration["last_activation"] == 'softmax':
            print(f"[NNFACTORY] Model compiled with softmax")
            self.instance_of_the_model.compile(optimizer='adam',
                                                loss='categorical_crossentropy',
                                                metrics=['accuracy'])
        else:
            print(f"[NNFACTORY] Model compiled with sigmoid")
            self.instance_of_the_model.compile(optimizer='adam',
                                                loss='binary_crossentropy',
                                                metrics=['accuracy'])

        # train
        print(f"[NNFACTORY] Fitting the model")
        if self.configuration["weight_samples"]:
            print(f"[NNFACTORY] Calculating the weights")

        history = self.instance_of_the_model.fit(X_train, y_train,
                                                    epochs=self.configuration["epochs"],
                                                    batch_size=self.configuration["batch_size"],
                                                    validation_data=(X_test, y_test) if X_test is not None else None,
                                                    verbose=self.configuration["verbose"],
                                                    sample_weight=compute_sample_weight("balanced", y_train) if self.configuration["weight_samples"] is True else None)

        if X_test is not None:

            # evaluate the model
            score = self.instance_of_the_model.evaluate(X_test, y_test, verbose=1)
            print(f"[SCORE worker {worker}] The score worker is {score[1] * 100}")

            if self.configuration["softmax"]:
                # predict
                prediction = self.instance_of_the_model.predict(X_test, verbose=1)
                prediction = np.argmax(prediction, axis=-1)

                # get true label
                true_labels = np.argmax(y_test, axis=-1)
            else:
                prediction = self.instance_of_the_model.predict(X_test)
                true_labels = y_test

            return [self.instance_of_the_model, (score[1] * 100), np.mean(history.history['accuracy']), true_labels, prediction]
        else:
            return [self.instance_of_the_model, None, np.mean(history.history['accuracy']), None, None]



