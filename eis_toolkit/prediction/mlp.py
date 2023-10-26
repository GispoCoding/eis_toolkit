from typing import Literal

import numpy
import numpy as np
import sklearn.base
from beartype import beartype
from sklearn.neural_network import MLPClassifier

from eis_toolkit.exceptions import InvalidArgumentTypeException, InvalidDatasetException
from eis_toolkit.prediction.model_performance_estimation import performance_model_estimation


@beartype
def crete_the_model(
    solver: str = "adam", alpha: float = 0.001, hidden_layer_sizes: tuple[int, int] = (16, 2), random_state: int = 0
) -> sklearn.base.BaseEstimator:
    """
    Do the model instantiation.

    Parameters:
        solver: this is what in keras is called optimizer.
        alpha: floating point represent regularization.
        hidden_layer_sizes: It represents the number of neurons in the ith hidden layer.
        random_state: random state for repeatability of results.

    Return:
        The instance of the compiled model.

    Raises:
        InvalidDatasetException: When the dataset is None.
    """

    # let's make an instance of classifier
    classifier = MLPClassifier(
        solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )

    return classifier


@beartype
def train_the_model(
    classifier: sklearn.base.BaseEstimator, train_dataset: numpy.ndarray, train_labels: numpy.ndarray
) -> sklearn.base.BaseEstimator:
    """
    Do the train the model.

    Parameters:
        classifier: An instance of sklearn BaseEstimator.
        train_dataset: Train features data.
        train_labels: Train labels data.


    Return:
        The instance of the compiled and fitted model.

    Raises:
        InvalidDatasetException: When the dataset is None.
    """

    if train_dataset is None or train_labels is None:
        raise InvalidDatasetException

    classifier.fit(train_dataset, train_labels)
    return classifier


@beartype
def evaluate_the_model(
    classifier: sklearn.base.BaseEstimator, test_dataset: numpy.ndarray, test_labels: numpy.ndarray
) -> float:
    """
    Do the evaluation of the model.

    Parameters:
        classifier: An instance of sklearn base estimator.
        test_dataset: Test data.
        test_labels: the test labels.
    Return:
        A float point number that shows the accuracy.

    Raises:
        InvalidDatasetException: When the dataset is None.

    """

    if test_dataset is None or test_labels is None:
        raise InvalidDatasetException

    # score
    score = classifier.score(test_dataset, test_labels)
    return score


@beartype
def predict_the_model(
    classifier: sklearn.base.BaseEstimator,
    test_dataset: numpy.ndarray,
    is_class_probability: bool = False,
    threshold_probability: float = None,
) -> numpy.ndarray:
    """
    Do the predictions of the model.

    Parameters:
        classifier: An instance of sklearn base estimator.
        test_dataset: the dataset to test.
        is_class_probability: if True the code return probability, otherwise it returns class.
        threshold_probability: works only if is_class_probability is True, is thresholds of probability.

    Return:
    A Numpy array with prediction (class if is_class_probability is set to false otherwise it returns probability).

    Raises:
        InvalidDatasetException: When the dataset is None.
        InvalidArgumentTypeException when the function try to make probability and the threshold is None.
    """

    if is_class_probability is not False and threshold_probability is None:
        raise InvalidArgumentTypeException

    if test_dataset is None:
        raise InvalidDatasetException

    # assign to classifier and data a vars I do not like see to much indexing
    if not is_class_probability:
        # predict class
        prediction = classifier.predict(test_dataset)
    else:
        # predict proba
        prediction = classifier.predict_proba(test_dataset)
        # assign proba to threshold
        prediction[prediction >= threshold_probability] = 1
    return prediction


@beartype
def mlp_train_evaluate_and_predict(
    dataset: np.ndarray,
    labels: np.ndarray,
    cross_validation_type: Literal["LOOCV", "KFOLD", "SKFOLD"],
    number_of_split: int,
    is_class_probability: bool = False,
    threshold_probability: float = None,
    is_predict_full_map: bool = False,
    solver: str = "adam",
    alpha: float = 0.001,
    hidden_layer_sizes: tuple[int, int] = (16, 2),
    random_state: int = 0,
) -> np.ndarray:
    """
    Do the training - evaluation - predictions steps with MLP.

    Parameters:
        dataset: Features data.
        labels: Labels data.
        cross_validation_type: selected cross validation method.
        number_of_split: number of split to divide the dataset.
        is_class_probability: if True the code return probability, otherwise it returns class.
     is_predict_full_map: if True the function will predict the full dataset otherwise predict only the test fold.
        threshold_probability: works only if is_class_probability is True, is thresholds of probability.
        solver: this is what in keras is called optimizer.
        alpha: floating point represent regularization.
        hidden_layer_sizes: It represents the number of neurons in the ith hidden layer.
        random_state: random state for repeatability of results.

    Return:
    A Numpy array with prediction (class if is_class_probability is set to false otherwise it returns probability).

    Raises:
        InvalidDatasetException: When the dataset is None.
        InvalidArgumentTypeException when the function try to make probability and the threshold is None.
    """

    # I need two local vars
    best_score = 0
    best_handler_list = list()

    if is_class_probability is not False and threshold_probability is None:
        raise InvalidArgumentTypeException

    if dataset is None or labels is None:
        raise InvalidDatasetException

    # select the cross validation method you need
    selected_cross_validation = performance_model_estimation(
        cross_validation_type=cross_validation_type, number_of_split=number_of_split
    )
    # start the training process
    for fold_number, (train_index, test_index) in enumerate(selected_cross_validation.split(dataset, labels)):

        # let's make an instance of classifier
        classifier = MLPClassifier(
            solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
        )

        # train the classifier
        classifier.fit(dataset[train_index], labels[train_index])
        # score
        fold_score = classifier.score(dataset[test_index], labels[test_index])

        if fold_number == 0:
            best_score = fold_score
            best_handler_list = [classifier, dataset[test_index]]
        else:
            if best_score < fold_score:
                best_score = fold_score
                best_handler_list = [classifier, dataset[test_index]]

    # assign to classifier and data a vars I do not like see to much indexing
    classifier = best_handler_list[0]

    if not is_predict_full_map:
        data = best_handler_list[1]
    else:
        data = dataset

    if not is_class_probability:
        # predict class
        prediction = classifier.predict(data)
    else:
        # predict proba
        prediction = classifier.predict_proba(data)
        # assign proba to threshold
        prediction[prediction >= threshold_probability] = 1

    return prediction
