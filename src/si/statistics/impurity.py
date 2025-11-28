import numpy as np


def entropy_impurity(y: np.ndarray) -> float:
    """
    Calculates the impurity of a dataset using entropy.

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    classes, counts = np.unique(y, return_counts=True)
    impurity = 0
    for i in range(len(classes)):
        impurity -= (counts[i] / len(y)) * np.log2(counts[i] / len(y))
    return impurity


def gini_impurity(y: np.ndarray) -> float:
    """
    Calculates the impurity of a dataset using the Gini index.

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    classes, counts = np.unique(y, return_counts=True)
    impurity = 1
    for i in range(len(classes)):
        impurity -= (counts[i] / len(y)) ** 2
    return impurity
