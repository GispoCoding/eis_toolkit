import matplotlib.pyplot as plt
import numpy as np
import pytest

from eis_toolkit.evaluation.plot_confusion_matrix import plot_confusion_matrix
from eis_toolkit.exceptions import InvalidDataShapeException


def test_plot_confusion_matrix():
    """Tests that plotting confusion matrix works as expected."""
    arr = np.array([[23, 5], [3, 30]])
    ax = plot_confusion_matrix(arr)
    assert isinstance(ax, plt.Axes)


def test_plot_confusion_matrix_invalid_matrix():
    """Tests that invalid input for plot confusion matrix raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        arr = np.array([[23, 5, 4], [3, 30, 2]])
        plot_confusion_matrix(arr)
