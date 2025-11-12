import numpy as np

from pytest import raises

from src.stats import perf_measure


def test_perf_measure():
    y_actual = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_hat = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    assert TP == 4
    assert FP == 1
    assert TN == 4
    assert FN == 1

    y_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_hat = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    assert TP == 0
    assert FP == 5
    assert TN == 5
    assert FN == 0

    y_actual = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_hat = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    assert TP == 3
    assert FP == 2
    assert TN == 2
    assert FN == 3

    y_actual = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_hat = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    assert TP == 0
    assert FP == 4
    assert TN == 0
    assert FN == 6


def test_perf_measure_errors():
    y_actual = np.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_hat = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    with raises(ValueError):
        perf_measure(y_actual, y_hat)

    y_actual = np.array([0, 0, 0])
    y_hat = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    with raises(ValueError):
        perf_measure(y_actual, y_hat)

    y_actual = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])
    y_hat = np.array([[1, 0, 1],
                      [0, 0, 1],
                      [1, 0, 1]])

    with raises(ValueError):
        perf_measure(y_actual, y_hat)
