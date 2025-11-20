from typing import Tuple
import numpy as np
import scipy

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Perform one-way ANOVA (F-test) across feature groups defined by target classes.

    This function splits the dataset features into groups based on the target classes,
    then computes the one-way ANOVA F-statistic and p-value for each feature across these groups.
    The ANOVA test is used to determine whether there are statistically significant differences
    between the means of three or more independent groups.

    Args:
        dataset (Dataset): A dataset object containing features (X) and target labels (y).
            The object must have the following attributes:
            - X: 2D array-like of shape (n_samples, n_features), the feature matrix.
            - y: 1D array-like of shape (n_samples,), the target labels.
            - get_classes(): A method that returns the unique class labels in the dataset.

    Returns:
        tuple: A tuple containing the F-statistic and p-value for each feature, as returned by
            `scipy.stats.f_oneway`. The tuple structure is (F-statistic, p-value).

    Example:
        >>> from scipy import stats
        >>> result = f_classification(my_dataset)
        >>> f_statistic, p_value = result
        >>> print(f"F-statistic: {f_statistic}, p-value: {p_value}")

    Notes:
        - The function assumes that `dataset.y` contains integer or categorical class labels.
        - If there are fewer than 2 classes, the ANOVA test is not meaningful and may raise an error.
        - The function uses `scipy.stats.f_oneway` under the hood.
    """
    classes = dataset.get_classes()
    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[mask, :]
        groups.append(group)
    return scipy.stats.f_oneway(*groups)
