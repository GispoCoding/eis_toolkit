import numpy as np
import tensorflow as tf

from eis_toolkit.bayesian_nn.bayesian_nn import generate_predictions, negative_loglikelihood

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


def test_compilation_of_the_model():
    """Do the first test."""
    predicted_dictionary = generate_predictions(
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
    assert len(predicted_dictionary) > 0
