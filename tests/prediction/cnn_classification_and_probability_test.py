import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from eis_toolkit.prediction.cnn_classification_and_probability import (
    train_and_predict_for_classification,
    train_and_predict_for_regression,
)
from eis_toolkit.prediction.model_performance_estimation import performance_model_estimation
from eis_toolkit.transformations.normalize_data import normalize_the_data
from eis_toolkit.transformations.one_hot_encoding import one_hot_encode


def test_do_the_classification():
    """Test for classification."""
    print(f"{os.getcwd()}")
    data = np.load(f'{os.path.join("data", "data.npy")}')
    labels = np.load(f'{os.path.join("data", "labels.npy")}')

    # do the encoding
    encoded_labels = one_hot_encode(labels, sparse_output=False)

    # create a scaler agent
    scaler_agent = StandardScaler()
    scaler_agent.fit(data.reshape(-1, data.shape[-1]))

    # make cv
    selected_cv = performance_model_estimation(cross_validation_type="SKFOLD", number_of_split=5)

    stacked_true, stacked_predicted = None, None

    for i, (train_idx, validation_idx) in enumerate(selected_cv.split(data, labels)):

        x_train = normalize_the_data(scaler_agent=scaler_agent, data=data[train_idx])
        y_train = encoded_labels[train_idx]

        x_validation = normalize_the_data(scaler_agent=scaler_agent, data=data[validation_idx])
        y_validation = encoded_labels[validation_idx]

        cnn_model, true_labels, predicted_labels, score = train_and_predict_for_classification(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            batch_size=32,
            epochs=10,
            conv_list=[4, 8],
            neuron_list=[8],
            input_shape_for_cnn=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            convolutional_kernel_size=(x_train.shape[3], x_train.shape[3]),
        )

        if stacked_true is None:
            stacked_true = np.argmax(true_labels, axis=1)
        else:
            stacked_true = np.concatenate((stacked_true, np.argmax(true_labels, axis=1)))

        if stacked_predicted is None:
            stacked_predicted = np.argmax(predicted_labels, axis=1)
        else:
            stacked_predicted = np.concatenate((stacked_predicted, np.argmax(predicted_labels, axis=1)))

    # make cm
    cm = confusion_matrix(stacked_true, stacked_predicted)
    print(cm)
    assert cm.shape[0] != 0 and cm.shape[1] != 0


def test_do_the_regression():
    """Test for classification."""
    print(f"{os.getcwd()}")
    data = np.load(f'{os.path.join("data", "data.npy")}')
    labels = np.load(f'{os.path.join("data", "labels.npy")}')

    # create a scaler agent
    scaler_agent = StandardScaler()
    scaler_agent.fit(data.reshape(-1, data.shape[-1]))

    # make cv
    selected_cv = performance_model_estimation(cross_validation_type="SKFOLD", number_of_split=5)

    stacked_true, stacked_predicted = None, None

    for i, (train_idx, validation_idx) in enumerate(selected_cv.split(data, labels)):

        x_train = normalize_the_data(scaler_agent=scaler_agent, data=data[train_idx])
        y_train = labels[train_idx]

        x_validation = normalize_the_data(scaler_agent=scaler_agent, data=data[validation_idx])
        y_validation = labels[validation_idx]

        cnn_model, true_labels, predicted_labels, probabilities, score = train_and_predict_for_regression(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            batch_size=32,
            epochs=10,
            conv_list=[4, 8],
            neuron_list=[8],
            input_shape_for_cnn=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            convolutional_kernel_size=(x_train.shape[3], x_train.shape[3]),
            threshold=0.5,
        )

        if stacked_true is None:
            stacked_true = true_labels
            print(stacked_true.shape)
        else:
            stacked_true = np.concatenate((stacked_true, true_labels))

        if stacked_predicted is None:
            stacked_predicted = predicted_labels
            print(predicted_labels.shape)
        else:
            stacked_predicted = np.concatenate((stacked_predicted, predicted_labels))

    # make cm
    cm = confusion_matrix(stacked_true, stacked_predicted)
    print(cm)
    assert cm.shape[0] != 0 and cm.shape[1] != 0
