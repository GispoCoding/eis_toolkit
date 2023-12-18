import numpy as np

from eis_toolkit.prediction.cnn_classification_and_regression import (
    train_and_predict_for_classification,
    train_and_predict_for_regression,
)


def test_do_classification():
    """Test for classification."""
    X = np.load("data/data.npy")
    y = np.load("data/labels.npy")

    best_model, cm = train_and_predict_for_classification(
        X=X,
        y=y,
        batch_size=32,
        epochs=1,
        cross_validation="LOOCV",
        input_shape_for_cnn=(X.shape[1], X.shape[2], X.shape[3]),
        convolutional_kernel_size=(X.shape[3], X.shape[3]),
        conv_list=[8, 16],
        neuron_list=[8],
        dropout_rate=0.1,
    )

    print(cm)
    assert cm.to_numpy().shape[0] != 0 and cm.to_numpy().shape[1] != 0


def test_do_regression():
    """Test for regression."""
    X = np.load("data/data.npy")
    y = np.load("data/labels.npy")
    best_model, cm = train_and_predict_for_regression(
        X=X,
        y=y,
        batch_size=32,
        epochs=1,
        cross_validation="LOOCV",
        input_shape_for_cnn=(X.shape[1], X.shape[2], X.shape[3]),
        convolutional_kernel_size=(X.shape[3], X.shape[3]),
        conv_list=[8, 16],
        neuron_list=[8],
        dropout_rate=0.1,
        threshold=0.5,
    )
    print(cm)
    assert cm.to_numpy().shape[0] != 0 and cm.to_numpy().shape[1] != 0
