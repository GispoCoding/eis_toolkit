import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from eis_toolkit.exceptions import CanNotMakeCategoricalLabelException, InvalidInputShapeException
from eis_toolkit.mlp.factory_helper import factory_helper


class mlp_factory(factory_helper):
    """Build and train a multi-layer perceptron(MLP)."""

    def __init__(self, cnn_configuration_dict: dict, general_configuration: dict) -> None:
        """
        Do the construction of the method.

        Args:
            cnn_configuration_dict (dictionary): Settings for the convolution neural network.
            general_configuration (dictionary): Configuration settings that are used throughout the class.

        Returns:
            void: The constructor does not return anything.

        Raises:
            Nothing: The constructor does not raise any exceptions.
        """
        self.cnn_configuration_dict = cnn_configuration_dict
        self.general_configuration = general_configuration

        super().__init__(general_configuration=general_configuration)

    def make_the_one_hot_encoding(self, labels_to_transform: np.array) -> np.array:
        """
        Perform one hot encoding on a numpy array of labels.

        Args:
            labels_to_transform (Numpy array): The function takes a numpy array of labels as input.

        Returns:
            labels_to_transform (Numpy array): One hot encoded array of the same labels.

        Raises:
            CanNotMakeCategoricalLabelException: This exception is raised when the one hot encoding is not successful.
        """
        enc = OneHotEncoder(handle_unknown="ignore")
        temp = np.reshape(labels_to_transform, (-1, 1))
        labels_to_transform = enc.fit_transform(temp).toarray()

        if (labels_to_transform.sum(axis=1) - np.ones(labels_to_transform.shape[0])).sum() != 0:
            raise CanNotMakeCategoricalLabelException

        return labels_to_transform

    def _generate_dynamically_the_MLP(
        self,
        input_channel_1: tuple,
        rescaling: bool,
        dense_nodes: list,
        final_output: int,
        last_activation: str,
        l_2: float,
    ) -> tf.keras.Model:
        """
        Do a multi-layer perceptron(MLP) neural network modek dynamically.

        Args:
            input_channel_1 (tuple): It specifies the shape of the input layer of the MLP.
            rescaling (boolean): It specifies whether the input data should be rescaled or not.
            dense_nodes (list): It specifies the number of nodes in each dense layer of the MLP.
            final_output (integer): It specifies the number of nodes in the final output layer of the MLP.
            last_activation (string): It specifies the activation function of the final output layer of the MLP.
            l_2 (float): It specifies the L2 regularization parameter for the MLP.

        Returns:
            Model (tf.keras.model): The function creates a Keras Model object.

        Raises:
            Nothing: The function does not raise any exceptions.
        """

        input_layer = tf.keras.Input(shape=input_channel_1, name="input_1")

        if rescaling:
            body = tf.keras.layers.Rescaling(1.0 / 255)(input_layer)
            flatten = tf.keras.layers.Flatten(name="flatten_layer")(body)
        else:
            flatten = tf.keras.layers.Flatten(name="flatten_layer")(input_layer)

        if len(dense_nodes) > 0:
            for ctn, dense_unit in enumerate(dense_nodes):
                if ctn == 0:
                    classifier = tf.keras.layers.Dense(
                        dense_unit,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                        bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                    )(flatten)
                else:
                    classifier = tf.keras.layers.Dense(
                        dense_unit,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                        bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                    )(classifier)

            classifier = tf.keras.layers.Dense(
                final_output,
                activation=last_activation,
                kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                name="final_classifier",
            )(classifier)
        else:
            classifier = tf.keras.layers.Dense(
                final_output,
                activation=last_activation,
                kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                name="final_classifier",
            )(flatten)

        model = tf.keras.Model(inputs=input_layer, outputs=classifier, name="model_1")
        return model

    def do_the_fitting_with_microprocessing(
        self,
        worker: int,
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        weighted_samples: bool,
    ) -> list:
        """
        Fits an MLP model on the training data and evaluates it's performance on the test data.

        Args:
            worker (integer): It indentifies the worker process that is executing this function.
            X_train (Numpy array): It contains the training data.
            y_train (Numpy array): It contains the training labels.
            X_test (Numpy array): It contains the test data.
            y_test (Numpy array): It contains the test labels.
            weighted_samples (boolean): It specifies whether to use class weights to balance the training data or not.

        Returns:
            The function returns a list of the following items:
            copiled_model (tf.keras.model): The function returns the compiled model.
            score (float): The function returns the accuracy of the model on the test data, multiplied by 100.
            history (tf.keras.callbacks.History): Mean of the history accuracy of the model training.
            true_labels (Numpy array): The function returns the true labels of the test data.
            prediction (Numpy array): The function returns the predicted labels of the test data.

        Raises:
            InvalidInputShapeException: The function first checks if the input data das a valid shape by ensuring
                                        that the number of features is greater than zero. If the shape is invalid,
                                        the function raises this exception.
        """

        if X_train.shape[1] <= 0 or X_test.shape[1] <= 0:
            raise InvalidInputShapeException

        self.cnn_configuration_dict["input_channel_1"] = X_train.shape[1]

        compiled_model = self._generate_dynamically_the_MLP(**self.cnn_configuration_dict)
        compiled_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = compiled_model.fit(
            X_train,
            y_train,
            epochs=self.general_configuration["epochs"],
            batch_size=self.general_configuration["batch_size"],
            validation_data=(X_test, y_test),
            verbose=self.general_configuration["verbose"],
            sample_weight=compute_sample_weight("balanced", y_train) if weighted_samples is True else None,
        )

        score = compiled_model.evaluate(X_test, y_test, verbose=1)

        if self.general_configuration["verbose"] > 0:
            print(f"[SCORE worker {worker}] The score worker is {score[1] * 100}")

        prediction = compiled_model.predict(X_test, verbose=1)
        prediction = np.argmax(prediction, axis=-1)
        true_labels = np.argmax(y_test, axis=-1)

        return [compiled_model, score[1] * 100, np.mean(history.history["accuracy"]), true_labels, prediction]

    def prepare_weights_for_pixelwise_classification(
        self, X_train: np.array, y_train: np.array, weighted_samples: bool
    ) -> list:
        """
        Do the weight for pixel-wise classification.

        Args:
            X_train (Numpy array): It contains the training data.
            y_train (Numpy array): It contains the training labels.
            weighted_samples (boolean): It specifies whether to use class weights to balance the training data or not.

        Returns:
            The function returns a list containing several items:
            compiled_model (tf.keras.model): The function returns the compiled model.
            None: An empty placeholder for the test score, which is not used in this case.
            history: Mean of the training accuracy.
            None: An empty placeholder for the true labels, which is not used in this case.
            None: An empty placeholder for the predicted labels, which is not used in this case.

        Raises:
            InvalidInputShapeException: This function first checks if the input data das a valid shape by ensuring
                                    the number of features is greater than zero. If the shape is invalid, the
                                    function raises this exception.
        """

        if X_train.shape[1] <= 0:
            raise InvalidInputShapeException

        self.cnn_configuration_dict["input_channel_1"] = X_train.shape[1]

        compiled_model = self._generate_dynamically_the_MLP(**self.cnn_configuration_dict)
        compiled_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = compiled_model.fit(
            X_train,
            y_train,
            epochs=self.general_configuration["epochs"],
            batch_size=self.general_configuration["batch_size"],
            verbose=self.general_configuration["verbose"],
            sample_weight=compute_sample_weight("balanced", y_train) if weighted_samples is True else None,
        )

        return [compiled_model, None, history, None, None]
