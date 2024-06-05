from numbers import Number

import numpy as np
from beartype.typing import Dict, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def summarize_label_metrics_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decimals: Optional[int] = None,
) -> Dict[str, Number]:
    """
    Generate a comprehensive report of various evaluation metrics for binary classification results.

    The output includes accuracy, precision, recall, F1 scores and confusion matrix elements
    (true negatives, false positives, false negatives, true positives).

    Args:
        y_true: True labels.
        y_pred: Predicted labels. The array should come from a binary classifier.
        decimals: Number of decimals used in rounding the scores. If None, scores are not rounded.
            Defaults to None.

    Returns:
        A dictionary containing the evaluated metrics.
    """
    metrics = {}

    accuracy = accuracy_score(y_true, y_pred)
    metrics["Accuracy"] = round(accuracy, decimals) if decimals is not None else accuracy

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    metrics["Precision"] = round(precision, decimals) if decimals is not None else precision
    metrics["Recall"] = round(recall, decimals) if decimals is not None else recall
    metrics["F1_score"] = round(f1, decimals) if decimals is not None else f1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["True_negatives"] = tn
    metrics["False_positives"] = fp
    metrics["False_negatives"] = fn
    metrics["True_positives"] = tp

    return metrics
