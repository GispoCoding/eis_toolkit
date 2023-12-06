import os

import numpy as np

from eis_toolkit.prediction.cnn_classification_and_regression import (
    train_and_predict_for_classification,
    train_and_predict_for_regression,
)

PATH = "Windows_test"

dataset = list()
labels = list()

for windows in os.listdir(PATH):
    path = f"{PATH}/{windows}"
    for folder in os.listdir(f"{path}"):
        for single_windows in os.listdir(f"{path}/{folder}"):
            full_path = f"{path}/{folder}/{single_windows}"
            dataset.append(np.load(full_path))
            temp = full_path.split("_")[-1].split(".")[0]
            labels.append(int(temp))

X = np.array(dataset)
y = np.array(labels)


def test_do_classification():
    """Test for classification."""
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
