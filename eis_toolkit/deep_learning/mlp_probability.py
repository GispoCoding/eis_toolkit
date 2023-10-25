import numpy as np
from sklearn.neural_network import MLPClassifier

from eis_toolkit.exceptions import InvalidDatasetException
from eis_toolkit.model_performance_estimation.model_performance_estimation import performance_model_estimation


def train_evaluate_predict_with_mlp(
    dataset: np.ndarray,
    labels: np.ndarray,
    cross_validation_type: str,
    number_of_split: int,
    is_class_probability: bool = False,
    threshold_probability: float = None,
    is_predict_full_map: bool = False,
    solver: str = "adam",
    alpha: float = 0.001,
    hidden_layer_sizes: tuple[int, int] = (16, 2),
    random_state=0,
) -> np.ndarray:
    """
    Do the training - evaluation - predictions steps with MLP.

    Parameters:
        dataset: Features data.
        labels: Labels data.
        cross_validation_type: selected cross validation method.
        number_of_split: number of split to divide the dataset.
        is_class_probability: if True the code return probability, otherwise it return class.
        is_predict_full_map: if True the function will predict the full dataset otherwise predict only the te4st fold.
        threshold_probability: works only if is_class_probability is True, is thresholds of probability.
        solver: this is what in keras is called optimizer.
        alpha: floating point represent regularization.
        hidden_layer_sizes: It represents the number of neurons in the ith hidden layer.
        random_state: random state for repeatability of results.
    Return:
        a numpy array with prediction (class if is_class_probability is set to false otherwise it return probability).
    Raises:
        InvalidDatasetException: When the dataset is None..
    """

    # I need two local vars
    best_score = 0
    best_handler_list = list()

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
