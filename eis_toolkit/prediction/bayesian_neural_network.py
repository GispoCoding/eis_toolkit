import os
import warnings
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
from beartype import beartype
from beartype.typing import Callable, List, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from eis_toolkit.exceptions import (
    InsufficientClassesException,
    InvalidDataShapeException,
    InvalidParameterValueException,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

tfd = tfp.distributions
tf.config.run_functions_eagerly(False)


class BayesianNeuralNetworkBase(BaseEstimator, ABC):
    """Base class for Bayesian Neural Networks."""

    @beartype
    def __init__(
        self,
        hidden_units=None,
        learning_rate=0.001,
        epochs=100,
        batch_size=512,
        n_samples=50,
        clip_norm=1.0,
        init_std=3.0,
        validation_split=0.0,
        early_stopping_patience=None,
        early_stopping_monitor="auto",
        early_stopping_min_delta=0.0,
        random_state=None,
        shuffle=True,
        stratified=None,
    ):
        """
        Initialize Bayesian Neural Network.

        Args:
            hidden_units: Number of units in each hidden layer. If None, it will be automatically
                set to [2 * n_features, n_features] in the fit() method.
            learning_rate: Learning rate for Adam optimizer. Values must be > 0. Defaults to 0.001.
            epochs: Maximum number of training epochs. Values must be >= 1. Defaults to 100.
            batch_size: Batch size for training. Values must be >= 1. Defaults to 512.
            n_samples: Number of Monte Carlo samples for predictions. Values must be >= 1. Defaults to 50.
            clip_norm: Gradient clipping norm. If None, no clipping is applied. Defaults to 1.0.
            init_std: Standard deviation of the prior distribution for the weights.
            validation_split: Fraction of training data to use for validation.
                Values must be between 0 and 1. Defaults to 0.1.
            early_stopping_patience: Number of epochs with no improvement after which training stops.
                If None, early stopping is disabled. Defaults to None.
            early_stopping_monitor: Metric to monitor for early stopping.
                Must be one of: "auto", "loss", "val_loss".
            early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            random_state: Seed for random number generation. Defaults to None.
            shuffle: Whether to shuffle training data before each epoch. Defaults to True.
                Should only be disabled in case of
                - Modeling with time series data
                - Other data that have an ordered structure and must be handled sequentially
            stratified: Whether to use stratified shuffling when splitting into training and validation data.
                Defaults to None (auto mode).
                For binary classification, it will be activated if the class ratio is below 1:3
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.clip_norm = clip_norm
        self.prior_std = init_std
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = early_stopping_min_delta
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratified = stratified

        # Internal attributes (set during fit)
        self.network_layers = None
        self.input_shape = None
        self.history = None
        self._is_fitted = False

        if random_state is not None:
            self._set_random_seed(random_state)

    @staticmethod
    def _set_random_seed(seed: int) -> None:
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def _initializer(shape: Tuple[int, int]) -> tf.Tensor:
        """Xavier initialization for weights."""
        return tf.random.truncated_normal(shape, mean=0.0, stddev=np.sqrt(2 / sum(shape)))

    def _create_layer_params(
        self,
        d_in: int,
        d_out: int,
    ) -> Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]:
        """Create variational parameters for a dense layer."""
        w_loc = tf.Variable(self._initializer((d_in, d_out)), name="w_loc")
        b_loc = tf.Variable(self._initializer((1, d_out)), name="b_loc")

        prior_rho = tf.math.log(tf.math.exp(self.prior_std) - 1)
        w_rho = tf.Variable(self._initializer((d_in, d_out)) - prior_rho, name="w_rho")
        b_rho = tf.Variable(self._initializer((1, d_out)) - prior_rho, name="b_rho")
        return w_loc, w_rho, b_loc, b_rho

    @staticmethod
    def _dense_layer_forward(
        x: tf.Tensor,
        params: Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable],
        training: bool = True,
        sampling: bool = False,
    ) -> tf.Tensor:
        """Forward pass through a Bayesian dense layer."""
        w_loc, w_rho, b_loc, b_rho = params
        w_sigma = tf.nn.softplus(w_rho)
        b_sigma = tf.nn.softplus(b_rho)

        if training:
            # Rademacher sampling for training
            s = tfp.random.rademacher(tf.shape(x))
            w_r = tfp.random.rademacher([tf.shape(x)[0], tf.shape(w_loc)[1]])
            b_r = tfp.random.rademacher([tf.shape(x)[0], tf.shape(b_loc)[1]])

            w_eps = tf.random.normal(tf.shape(w_loc))
            b_eps = tf.random.normal(tf.shape(b_loc))

            w_samples = w_sigma * w_eps
            b_samples = b_sigma * b_eps

            w_perturb = w_r * tf.matmul(x * s, w_samples)
            w_outputs = tf.matmul(x, w_loc) + w_perturb
            b_outputs = b_loc + b_r * b_samples

            return w_outputs + b_outputs
        elif sampling:
            # Standard MC sampling for prediction
            w_eps = tf.random.normal(tf.shape(w_loc))
            b_eps = tf.random.normal(tf.shape(b_loc))
            w = w_loc + w_sigma * w_eps
            b = b_loc + b_sigma * b_eps
            return tf.matmul(x, w) + b
        else:
            # Mean prediction
            return tf.matmul(x, w_loc) + b_loc

    # @staticmethod
    def _kl_divergence(
        self,
        params: Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable],
    ) -> tf.Tensor:
        """Calculate KL divergence between variational posterior and prior."""
        w_loc, w_rho, b_loc, b_rho = params

        weight_posterior = tfd.Normal(w_loc, tf.nn.softplus(w_rho))
        bias_posterior = tfd.Normal(b_loc, tf.nn.softplus(b_rho))
        prior = tfd.Normal(0.0, self.prior_std)

        kl_w = tf.reduce_sum(tfd.kl_divergence(weight_posterior, prior))
        kl_b = tf.reduce_sum(tfd.kl_divergence(bias_posterior, prior))

        return kl_w + kl_b

    def _build_network_with_prior(
        self,
        input_shape: int,
        hidden_units: List[int],
        output_dim: int = 1,
    ) -> List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]]:
        """Build network layers with given dimensions."""
        all_units = [input_shape] + hidden_units + [output_dim]
        input_dims = all_units[:-1]
        output_dims = all_units[1:]
        return [self._create_layer_params(d_in, d_out) for d_in, d_out in zip(input_dims, output_dims)]

    def _forward_layers(
        self,
        x: tf.Tensor,
        layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]],
        training: bool = True,
        sampling: bool = False,
    ) -> tf.Tensor:
        """Forward pass through multiple layers."""
        for i, params in enumerate(layers):
            x = self._dense_layer_forward(x, params, training, sampling)
            if i < len(layers) - 1:
                x = tf.nn.relu(x)
        return x

    def _calculate_total_kl(self, layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]]) -> tf.Tensor:
        """Calculate total KL divergence across all layers."""
        return tf.add_n([self._kl_divergence(layer) for layer in layers])

    @staticmethod
    def _welford_update(
        new_value: np.ndarray,
        count: int,
        running_mean: np.ndarray,
        M2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Welford's online algorithm update step.

        Args:
            new_value: New observation.
            count: Current sample count.
            running_mean: Current mean.
            M2: Current sum of squared differences (M2).

        Returns:
            Updated mean and M2.
        """
        delta_before_update = new_value - running_mean
        running_mean += delta_before_update / count
        delta_after_update = new_value - running_mean
        M2 += delta_before_update * delta_after_update

        return running_mean, M2

    def _create_train_step(
        self,
        network_layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]],
        loss_function: Callable,
        optimizer: tf.keras.optimizers.Optimizer,
        clip_norm: Optional[Number],
    ) -> Callable:
        """Create training step function with specified loss."""

        @tf.function
        def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = self._forward_layers(x_batch, network_layers, training=True, sampling=False)

                # Losses
                kl_loss = self._calculate_total_kl(network_layers)
                likelihood_loss = loss_function(y_batch, y_pred)
                elbo_loss = (kl_loss / N) + likelihood_loss

            # Gradients
            variables = [v for layer in network_layers for v in layer]
            gradients = tape.gradient(elbo_loss, variables)

            if clip_norm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)

            optimizer.apply_gradients(zip(gradients, variables))

            return elbo_loss

        return train_step

    def _create_val_step(
        self,
        network_layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]],
        loss_function: Callable,
    ) -> Callable:
        """Create validation step function with specified loss."""

        @tf.function
        def val_step(x_batch: tf.Tensor, y_batch: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
            y_pred = self._forward_layers(x_batch, network_layers, training=False, sampling=False)

            kl_loss = self._calculate_total_kl(network_layers)
            likelihood_loss = loss_function(y_batch, y_pred)
            val_elbo_loss = (kl_loss / N) + likelihood_loss

            return val_elbo_loss

        return val_step

    @staticmethod
    def _validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input data shapes.

        Args:
            X: Training data
            y: Target values

        Returns:
            None

        Raises:
            InvalidDataShapeException: If X or y have invalid shapes.
        """
        if X.ndim != 2:
            raise InvalidDataShapeException(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise InvalidDataShapeException(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise InvalidDataShapeException(f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}")

        return None

    def _check_stratified(self, y: np.ndarray) -> bool:
        """Check if stratified sampling should be used (or not)."""
        if self.shuffle and self.stratified is not None:
            return self.stratified

        if not isinstance(self, ClassifierMixin):
            return False

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 2:
            minority_ratio = counts.min() / len(y)
            return minority_ratio < 0.3

        return False

    def _create_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset with consistent preprocessing."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(X.shape[0])

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE).cache()

    def _prepare_datasets(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.Tensor, Optional[tf.Tensor]]:
        """
        Prepare training/validation datasets from validated data.

        In case of a validation split, the training dataset will be split into two parts:
        - The split is performed using train_test_split with the specified validation_split ratio
        - If shuffle is True for the model, data will be shuffled before each training epoch
        - Training data: first portion [:split]
        - Validation data: remaining portion [split:]

        Args:
            X: Validated training data
            y: Validated target values

        Returns:
            Tuple of (train_dataset, val_dataset, N_train, N_val)
        """
        X, y = X.astype(np.float32), y.astype(np.float32)

        if self.validation_split > 0:
            use_stratified = self._check_stratified(y)

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_split,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=y if use_stratified else None,
            )

            train_dataset = self._create_dataset(X_train, y_train, shuffle=self.shuffle)
            val_dataset = self._create_dataset(X_val, y_val, shuffle=False)

            N_train = tf.constant(X_train.shape[0], dtype=tf.float32)
            N_val = tf.constant(X_val.shape[0], dtype=tf.float32)

            return train_dataset, val_dataset, N_train, N_val
        else:
            train_dataset = self._create_dataset(X, y, shuffle=self.shuffle)
            N_train = tf.constant(X.shape[0], dtype=tf.float32)

            return train_dataset, None, N_train, None

    @abstractmethod
    def _get_loss_function(self):
        """
        Get the loss function for this model type.

        Returns:
            Callable loss function.
        """
        pass

    @abstractmethod
    def _get_output_dim(self, y: np.ndarray) -> int:
        """
        Get the output dimension for this model type.

        Returns:
            int: Output dimension.
        """
        pass

    @abstractmethod
    def _process_predictions(self, logits: tf.Tensor) -> tf.Tensor:
        """
        Transform raw logits to the prediction space.

        For classification: apply sigmoid/softmax
        For regression: return as-is or apply custom transformation

        Args:
            Raw network outputs

        Returns:
            Transformed predictions
        """
        pass

    def _determine_hidden_units(self, n_features: int):
        """Determine hidden layer architecture based on input features.

        Args:
            n_features: Number of input features from X.

        Returns:
            Depth and width information for hidden layers.
            Defaults are 2 layers [2*n, n] with a minimum of [32, 16].
        """
        if self.hidden_units is None:
            unit_1 = 2 * n_features
            unit_2 = n_features
            return [unit_1, unit_2]
        else:
            return self.hidden_units

    def _validate_parameters(self):
        """
        Validate all input parameters.

        Raises:
            InvalidParameterValueException: If any parameter has invalid values.
        """
        if self.learning_rate <= 0:
            raise InvalidParameterValueException("Learning rate must be greater than 0.")

        if self.epochs < 1:
            raise InvalidParameterValueException("Number of epochs must be at least 1.")

        if self.batch_size < 1:
            raise InvalidParameterValueException("Batch size must be at least 1.")

        if self.n_samples < 1:
            raise InvalidParameterValueException("Number of Monte Carlo samples must be at least 1.")

        if not (0 <= self.validation_split <= 1):
            raise InvalidParameterValueException("Validation split must be between 0 and 1.")

        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise InvalidParameterValueException("Early stopping patience must be at least 1.")

        if self.early_stopping_min_delta < 0:
            raise InvalidParameterValueException("Early stopping min_delta must be >= 0.")

        if self.hidden_units is not None:
            if len(self.hidden_units) == 0:
                raise InvalidParameterValueException("Hidden units list must not be empty.")

            if not all(isinstance(unit, int) and unit > 0 for unit in self.hidden_units):
                raise InvalidParameterValueException("All hidden units must be positive integers.")

    def _determine_early_stopping_monitor(self) -> Optional[str]:
        """
        Determine which metric to monitor for early stopping.

        Returns:
            The metric name to monitor, or None if early stopping should be disabled
        """
        use_validation = self.validation_split > 0
        monitor = self.early_stopping_monitor

        if monitor == "loss":
            return "loss"

        if monitor == "auto":
            return "val_loss" if use_validation else "loss"

        if monitor == "val_loss":
            if use_validation:
                return "val_loss"
            else:
                warnings.warn(
                    "early_stopping_monitor='val_loss' but validation_split=0." "Early stopping will be disabled.",
                    UserWarning,
                )
                return None

        # Invalid monitor value
        raise InvalidParameterValueException(
            f"Invalid early_stopping_monitor '{monitor}'. " "Must be one of: 'auto', 'loss', 'val_loss'"
        )

    @staticmethod
    def _create_metrics(use_validation: bool) -> dict[str, tf.keras.metrics.Metric]:
        """Create the history for tracking training progress."""
        metrics = {"loss": tf.keras.metrics.Mean(name="loss")}

        if use_validation:
            metrics["val_loss"] = tf.keras.metrics.Mean(name="val_loss")

        return metrics

    @staticmethod
    def _reset_metrics(metrics: dict[str, tf.keras.metrics.Metric]) -> None:
        """Reset all metrics to the initial state."""
        for metric in metrics.values():
            metric.reset_states()

    @staticmethod
    def _update_history(
        history: dict[str, List[float]], metrics: dict[str, tf.keras.metrics.Metric]
    ) -> dict[str, float]:
        """Update the training history with current epoch results."""
        epoch_results = {}

        for name, metric in metrics.items():
            value = metric.result().numpy()
            history[name].append(value)
            epoch_results[name] = value

        return epoch_results

    @beartype
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianNeuralNetworkBase":
        """
        Train the Bayesian Neural Network.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self (fitted model).

        Raises:
            InvalidDataShapeException: If X or y have invalid shapes.
            InvalidParameterValueException: If parameters have invalid values.
        """
        self._validate_parameters()
        self._validate_input_data(X, y)
        self.input_shape = X.shape[1]

        if self.validation_split > 0.5:
            warnings.warn(
                f"validation_split={self.validation_split} means "
                f"{self.validation_split * 100}% of data is used for validation. "
                "This is unusual and may lead to poor training performance.",
                UserWarning,
            )

        # Determine early stopping monitor
        monitor_metric = self._determine_early_stopping_monitor()

        # Prepare datasets
        use_validation = self.validation_split > 0
        train_dataset, val_dataset, N_train, N_val = self._prepare_datasets(X, y)

        # Build network and create training components
        hidden_units = self._determine_hidden_units(self.input_shape)
        output_dim = self._get_output_dim(y)
        self.network_layers = self._build_network_with_prior(self.input_shape, hidden_units, output_dim=output_dim)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_function = self._get_loss_function()

        train_step = self._create_train_step(self.network_layers, loss_function, optimizer, self.clip_norm)
        val_step = self._create_val_step(self.network_layers, loss_function) if use_validation else None

        # Initialize metrics and history
        metrics = self._create_metrics(use_validation)
        self.history = {key: [] for key in metrics.keys()}

        # Early stopping setup
        best_monitor_value = float("inf")
        patience_counter = 0

        # Training loop
        desc = f"Training {self.__class__.__name__}"
        epochs_range = tqdm(range(self.epochs), desc=desc, unit="epoch")

        for epoch in epochs_range:
            self._reset_metrics(metrics)

            # Training phase
            for x_batch, y_batch in train_dataset:
                loss = train_step(x_batch, y_batch, N_train)
                metrics["loss"].update_state(loss)

            # Validation phase
            if use_validation and val_step is not None:
                for x_batch, y_batch in val_dataset:
                    val_loss = val_step(x_batch, y_batch, N_val)
                    metrics["val_loss"].update_state(val_loss)

            # Update history
            epoch_results = self._update_history(self.history, metrics)

            # Update progress bar
            epochs_range.set_postfix({key: f"{value:.4f}" for key, value in epoch_results.items()})

            # Early stopping check
            if self.early_stopping_patience and monitor_metric:
                current_monitor_value = epoch_results[monitor_metric]
                improvement = best_monitor_value - current_monitor_value

                if improvement > self.early_stopping_min_delta:
                    best_monitor_value = current_monitor_value
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} (monitoring {monitor_metric})")
                    break

        self._is_fitted = True
        return self

    @beartype
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates using Monte Carlo sampling.

        Args:
            X: Data to predict on, shape (n_samples, n_features).

        Returns:
            Tuple of (predictions, uncertainties).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction!")

        X_data = tf.convert_to_tensor(X, dtype=tf.float32)

        # Get initial shape from a forward pass
        initial_forward_pass = self._forward_layers(X_data, self.network_layers, training=False, sampling=False)
        initial_transformed = self._process_predictions(initial_forward_pass)
        mean = np.zeros_like(initial_transformed)
        M2 = np.zeros_like(initial_transformed)

        for sample_idx in tqdm(range(self.n_samples), desc="Monte Carlo Sampling", unit="sample"):
            logits = self._forward_layers(X_data, self.network_layers, training=False, sampling=True)
            predictions = self._process_predictions(logits).numpy()

            # Welford's algorithm update
            mean, M2 = self._welford_update(predictions, sample_idx + 1, mean, M2)

        variance = M2 / self.n_samples if self.n_samples > 1 else np.zeros_like(mean)
        std_dev = np.sqrt(variance)

        return mean.ravel(), std_dev.ravel()


class BayesianNeuralNetworkClassifier(BayesianNeuralNetworkBase, ClassifierMixin):
    """Bayesian Neural Network Classifier for binary classification."""

    def _get_loss_function(self):
        """Get binary cross-entropy loss function."""

        def binary_crossentropy_loss(y_true, y_pred):
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(y_true, axis=-1), logits=y_pred)
            )

        return binary_crossentropy_loss

    def _get_output_dim(self, y: np.ndarray) -> int:
        """Determine the output dimension for classification."""
        self.n_classes = len(np.unique(y))

        if self.n_classes < 2:
            raise InsufficientClassesException(f"Classification requires at least 2 classes, got {self.n_classes}. ")

        return 1 if self.n_classes == 2 else self.n_classes

    def _process_predictions(self, logits: tf.Tensor) -> tf.Tensor:
        """Transform logits to probabilities using sigmoid."""
        if self.n_classes <= 2:
            return tf.nn.sigmoid(logits)
        else:
            return tf.nn.softmax(logits)

    @beartype
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for sklearn compatibility.

        Args:
            X: Data to predict on, shape (n_samples, n_features).

        Returns:
            Array of predicted class labels (0 or 1), shape (n_samples,).
        """
        probabilities, _ = self.predict_with_uncertainty(X)
        return (probabilities >= 0.5).astype(int).ravel()

    @beartype
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for sklearn compatibility.

        Args:
            X: Data to predict on, shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        probabilities, _ = self.predict_with_uncertainty(X)
        probabilities = probabilities.ravel()
        return np.column_stack([1 - probabilities, probabilities])


@beartype
def bayesian_neural_network_classifier_train(
    X: np.ndarray,
    y: np.ndarray,
    hidden_units: Optional[List[int]] = None,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 512,
    n_samples: int = 50,
    validation_split: float = 0.0,
    early_stopping_patience: Optional[int] = None,
    early_stopping_monitor: str = "auto",
    early_stopping_min_delta: float = 0.0,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    stratified: Optional[bool] = None,
) -> BayesianNeuralNetworkClassifier:
    """
    Train a Bayesian Neural Network classifier for binary classification.

    Args:
        X: Training data of shape (n_samples, n_features).
        y: Binary target labels of shape (n_samples,).
        hidden_units: List of hidden layer sizes. Defaults to None (auto mode).
        learning_rate: Learning rate for optimizer. Values must be > 0. Defaults to 0.001.
        epochs: Number of training epochs. Values must be >= 1. Defaults to 100.
        batch_size: Batch size for training. Values must be >= 1. Defaults to 512.
        n_samples: Number of Monte Carlo samples for prediction. Values must be >= 1. Defaults to 50.
        validation_split: Fraction of data to use for validation. If 0, no validation is used. Defaults to 0.
        early_stopping_patience: Number of epochs with no improvement to wait before stopping.
            If None, early stopping is disabled. Defaults to None.
        early_stopping_monitor: Metric to monitor for early stopping. Options: "auto", "loss", "val_loss".
            "auto" uses "val_loss" if validation_split > 0, otherwise "loss". Defaults to "auto".
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        random_state: Seed for random number generation. Defaults to 42 for reproducibility.
        shuffle: Whether to shuffle the training data before training. Defaults to True.
        stratified: Whether to use stratified sampling for training. Defaults to None (auto mode).

    Returns:
        The trained BayesianNeuralNetworkClassifier.
    """
    model = BayesianNeuralNetworkClassifier(
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_min_delta=early_stopping_min_delta,
        random_state=random_state,
        shuffle=shuffle,
        stratified=stratified,
    )
    return model.fit(X, y)
