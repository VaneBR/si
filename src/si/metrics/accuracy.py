import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        The accuracy of the model
    """
    # deal with predictions like [[0.52], [0.91], ...] and [[0.3, 0.7], [0.6, 0.4], ...]
    # they need to be in the same format: [0, 1, ...] and [1, 0, ...]
    def correct_format(y):
        if len(y[0]) == 1:
            corrected_y = [np.round(y[i][0]) for i in range(len(y))]
        else:
            corrected_y = [np.argmax(y[i]) for i in range(len(y))]
        return np.array(corrected_y)
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray):
        y_true = correct_format(y_true)
    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray):
        y_pred = correct_format(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)