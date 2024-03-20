from typing import Literal, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


def __posterior_mean_field(
    kernel_size: int or tuple[int, int],
    bias_size: int = 0,
    scale: float = 1e-5,
    reinterpreted_batch_ndims: int = 1,
    dtype=tf.float32,
) -> tf.keras.Sequential:
    """
    Do the posterior mean field.

    Parameters:
    - kernel_size: integer or a tuple of integer of the filter dimension.
    - bias_size: the size of the bias default parameters is zero.
    - dtype: datatype of the layer default is float 32.
    - scale: scale of the normal distribution.
    - reinterpreted_batch_ndims: the reinterpreted dimension of the batch.

    Returns:
    - a sequential model the posteriori distribution.
    """
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t[..., :n], scale=scale + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=reinterpreted_batch_ndims,
                )
            ),
        ]
    )


def __prior_trainable(
    kernel_size,
    bias_size=0,
    dtype=tf.float32,
    scale: float = 1.0,
    reinterpreted_batch_ndims: int = 1,
) -> tf.keras.Sequential:
    """
    Do the learns of the optimal parameter for the bayesian NN.

    Parameters:
    - kernel_size: integer or a tuple of integer of the filter dimension.
    - bias_size: the size of the bias default parameters is zero.
    - dtype: datatype of the layer default is float 32.
    - scale: scale of the normal distribution.
    - reinterpreted_batch_ndims: the reinterpreted dimension of the batch.

    Returns:
    - trainable model
    """
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=scale), reinterpreted_batch_ndims=reinterpreted_batch_ndims
                )
            ),
        ]
    )


def __create_probabilistic_bnn_model(
    train_size: int,
    hidden_units: list[int],
    features_name: list[str or int],
    last_activation: Literal["softmax", "sigmoid"],
) -> tf.keras.Model:
    """
    Do the bayesian model.

    Parameters:
    - train_size: the train size.
    - hidden_units: number of the layers of the network.
    - features_name: name of the feature to impute to the network.
    - last_activation: the output activation of the network.

    Returns:
    - the model before compilation
    """
    inputs = {}
    features = None
    for feature_name in features_name:
        inputs[feature_name] = tf.keras.layers.Input(name=feature_name, shape=(1,), dtype=tf.float32)

        for ctn, units in enumerate(hidden_units):
            features = (
                tfp.layers.DenseVariational(
                    units=units,
                    make_prior_fn=__prior_trainable,
                    make_posterior_fn=__posterior_mean_field,
                    kl_weight=1 / train_size,
                    activation=last_activation,
                )(features)
                if ctn != 0
                else (inputs)
            )

        distribution_params = tf.keras.layers.Dense(units=2)(features)
        outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


def negative_loglikelihood(targets: tf.Tensor, estimated_distribution: tf.Tensor) -> tf.Tensor:
    """
    Do the negative likelihood loss.

    Parameters:
    - targets: real labels needed to compute this loss.
    - estimated_distribution: the predicted labels needed to compute this loss.

    Returns:
    - negative log probabilities of the class
    """
    return -estimated_distribution.log_prob(targets)


def generate_predictions(
    train_dataset: Union[tf.data.Dataset, pd.DataFrame],
    test_dataset: Union[tf.data.Dataset, pd.DataFrame],
    features_name: list[str or int],
    last_activation: Literal["softmax", "sigmoid"],
    hidden_units: list[int],
    batch_size: int,
    num_epochs: int,
    optimizer: Union[
        tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD
    ],
    loss: Union[tf.keras.losses.BinaryCrossentropy, tf.keras.losses.CategoricalCrossentropy, negative_loglikelihood],
    metrics: Union[tf.keras.metrics.RootMeanSquaredError, tf.keras.metrics.Accuracy],
) -> list[dict[str, any]]:
    """
    Compute inferences and generate predictions with the bayesian model.

    Parameters:
    - train_dataset: the portion of the dataset used for training.
    - test_dataset: the portion of the dataset used for testing.
    - features_name: a list of features name or number.
    - last_activation: the output of the model.
    - hidden_units: the number of the networks layer.
    - batch_size: the batch size.
    - num_epochs: the number of epochs.
    - optimizer: optimizer of the network.
    - loss: measure the error between the predicted and true values.
    - metrics: performance of the model.
    Returns:
    - a list of dict that contains, predicted mean, std, CI lower and upper, the actual value.
    """

    # here create the probabilistic model
    prob_bnn_model = __create_probabilistic_bnn_model(
        train_size=len(train_dataset),
        hidden_units=hidden_units,
        features_name=features_name,
        last_activation=last_activation,
    )

    prob_bnn_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    prob_bnn_model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset, verbose=1)

    results = []
    sample = test_dataset.cardinality().numpy() * batch_size
    examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[0]
    prediction_distribution = prob_bnn_model(examples)

    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()

    for idx in range(sample):
        results.append(
            {
                "mean": round(prediction_mean[idx][0], 2),
                "stddev": round(prediction_stdv[idx][0], 2),
                "95% CI lower": round(lower[idx][0], 2),
                "95% CI upper": round(upper[idx][0], 2),
                "Actual": targets[idx].numpy(),
            }
        )

    return results
