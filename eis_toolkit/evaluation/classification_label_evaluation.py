from numbers import Number
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def summarize_label_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Number]:
    """
    Generate a comprehensive report of various evaluation metrics for binary classification results.

    The output includes accuracy, precision, recall, F1 scores and confusion matrix elements
    (true negatives, false positives, false negatives, true positives).

    Args:
        y_true: True labels.
        y_pred: Predicted labels. The array should come from a binary classifier.

    Returns:
        A dictionary containing the evaluated metrics.
    """
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["F1_score"] = f1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["True_negatives"] = tn
    metrics["False_positives"] = fp
    metrics["False_negatives"] = fn
    metrics["True_positives"] = tp

    return metrics
