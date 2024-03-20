import numpy as np
import tensorflow as tf

from eis_toolkit.bayesian_nn.bayesian_nn import generate_predictions, negative_loglikelihood

X_train = np.random.rand(1000, 5)
y_train = np.random.randint(0, 2, 1000)

feature_names = ["MAG_1", "MAG_2", "MAG_3", "MAG_4", "MAG_5"]
data_dict = {feature: X_train[:, idx] for idx, feature in enumerate(feature_names)}

X = tf.data.Dataset.from_tensor_slices((data_dict, y_train)).shuffle(1000).batch()


if __name__ == "__main__":

    print(X.shape)
    predicted_dictionary = generate_predictions(
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
