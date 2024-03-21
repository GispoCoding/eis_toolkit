import numpy as np
import pytest
import tensorflow as tf

from eis_toolkit.bayesian_nn.bayesian_nn import (
    generate_prediction_using_traditional_arrays,
    generate_predictions_with_tensor_api,
    negative_loglikelihood,
)
from eis_toolkit.exceptions import InvalidInputDataException

X_train = np.random.rand(1000, 5)
y_train = np.random.randint(0, 2, 1000)
X_test = np.random.rand(200, 5)
y_test = np.random.randint(0, 2, 200)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

feature_names = ["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"]
train_data_dict = {feature: X_train[:, idx] for idx, feature in enumerate(feature_names)}
test_data_dict = {feature: X_test[:, idx] for idx, feature in enumerate(feature_names)}


def test_the_numpy_inputs():
    """This test focus in running the baysian nn loaded with numpy arrays."""
    # convert to dict
    X_train = {}
    X_test = {}

    for el in feature_names:
        X_train[el] = np.random.rand(1000, 1)
        X_test[el] = np.random.rand(200, 1)

    pred = generate_prediction_using_traditional_arrays(
        X_train=X_train,
        y_train=y_train.astype("float"),
        X_test=X_test,
        y_test=y_test.astype("float"),
        validation_split=None,
        features_name=["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"],
        last_activation="sigmoid",
        hidden_units=[16, 8],
        batch_size=32,
        num_epochs=10,
        optimizer=tf.keras.optimizers.Adam(),
        loss=negative_loglikelihood,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )

    assert len(pred) > 0


def test_compilation_of_the_model():
    """This test focus on compilation of the model it checks if it returns results."""
    predicted_dictionary = generate_predictions_with_tensor_api(
        train_dataset=tf.data.Dataset.from_tensor_slices((train_data_dict, y_train.astype("float")))
        .shuffle(1000)
        .batch(32),
        test_dataset=tf.data.Dataset.from_tensor_slices((test_data_dict, y_test.astype("float")))
        .shuffle(200)
        .batch(32),
        features_name=["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"],
        last_activation="sigmoid",
        hidden_units=[16, 8],
        batch_size=32,
        num_epochs=10,
        optimizer=tf.keras.optimizers.Adam(),
        loss=negative_loglikelihood,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )

    assert len(predicted_dictionary) > 0 and isinstance(predicted_dictionary, list)


def test_returned_model():
    """This test focus on returning the model only for further analysis."""
    model = generate_predictions_with_tensor_api(
        train_dataset=tf.data.Dataset.from_tensor_slices((train_data_dict, y_train.astype("float")))
        .shuffle(1000)
        .batch(32),
        test_dataset=None,
        features_name=["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"],
        last_activation="sigmoid",
        hidden_units=[16, 8],
        batch_size=32,
        num_epochs=10,
        optimizer=tf.keras.optimizers.Adam(),
        loss=negative_loglikelihood,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )

    assert isinstance(model, tf.keras.Model)


def test_invalid_parameters_inputs_exception():
    """This test focus on throwing an exception when the data is None."""
    with pytest.raises(InvalidInputDataException):
        _ = generate_predictions_with_tensor_api(
            train_dataset=None,
            test_dataset=None,
            features_name=["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"],
            last_activation="sigmoid",
            hidden_units=[16, 8],
            batch_size=32,
            num_epochs=10,
            optimizer=tf.keras.optimizers.Adam(),
            loss=negative_loglikelihood,
            metrics=tf.keras.metrics.RootMeanSquaredError(),
        )
