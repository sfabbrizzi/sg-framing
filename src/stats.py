# general imports
import numpy as np

# typing
from typing import Tuple


def perf_measure(y_actual: np.array, y_hat: np.array) -> Tuple[int, int, int, int]:
    """Calculate performance measures for binary classification.

    Parameters
    ----------
    y_actual : np.array
        Ground truth (correct) labels.
    y_hat : np.array
        Predicted labels, as returned by a classifier.

    Returns
    -------
    Tuple[int, int, int, int]
        True positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).
    """
    if len(y_actual) != len(y_hat):
        raise ValueError("Length of actual and predicted labels must be the same.")
    if y_actual.ndim != 1 or y_hat.ndim != 1:
        raise ValueError("y_actual and y_hat must be 1-dimensional arrays.")
    if not set(np.unique(y_actual)).issubset({0, 1}) or not set(np.unique(y_hat)).issubset({0, 1}):
        raise ValueError("y_actual and y_hat must contain only binary values (0 and 1).")

    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    for i in range(len(y_hat)):
        if (y_actual[i] == 1) and (y_hat[i] == 1):
            TP += 1
        if (y_hat[i] == 1) and (y_actual[i] != y_hat[i]):
            FP += 1
        if (y_actual[i] == 0) and (y_hat[i] == 0):
            TN += 1
        if (y_hat[i] == 0) and (y_actual[i] != y_hat[i]):
            FN += 1

    return TP, FP, TN, FN
