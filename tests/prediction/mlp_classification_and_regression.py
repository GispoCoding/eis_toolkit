import numpy as np
import pandas as pd
import tensorflow as tf

from eis_toolkit.prediction.mlp_classification_and_regression import (
    train_and_predict_for_classification,
    train_and_predict_for_regression,
)

X = pd.read_csv("../data/remote/fake_smote_data.csv").to_numpy()
labels = np.random.randint(2, size=X.shape[0])

print(X.shape, labels.shape)


def test_classification_compile_and_produces_cm():
    """Do the test."""
    model_to_return, df = train_and_predict_for_classification(
        X=X,
        y=labels,
        batch_size=32,
        epochs=10,
        cross_validation="LOOCV",
        input_shape_for_mlp=(X.shape[1]),
        sample_weights=True,
        neuron_list=[16, 24, 32],
        dropout_rate=0.1,
        last_activation="softmax",
        regularization=None,
        data_augmentation=False,
        optimizer="Adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        output_units=len(np.unique(labels)),
    )

    assert df.to_numpy().shape[0] != 0 and df.to_numpy().shape[1] != 0


def test_regression_compile_and_produces_cm():
    """Do the test."""
    model_to_return, df = train_and_predict_for_regression(
        X=X,
        y=labels,
        batch_size=32,
        epochs=10,
        cross_validation="LOOCV",
        input_shape_for_mlp=(X.shape[1]),
        sample_weights=True,
        neuron_list=[16, 24, 32],
        dropout_rate=0.1,
        last_activation="sigmoid",
        regularization=None,
        data_augmentation=False,
        optimizer="Adam",
        threshold=0.5,
        loss=tf.keras.losses.BinaryCrossentropy(),
        output_units=1,
    )
    assert df.to_numpy().shape[0] != 0 and df.to_numpy().shape[1] != 0
