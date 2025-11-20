from typing import List

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset


def k_fold_cross_validation(model: Model, dataset: Dataset, 
                            scoring: callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model on each fold.
    """
    num_samples = dataset.X.shape[0]
    fold_size = num_samples // cv
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for fold in range(cv):
        # Determine the indices for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        # Split the data into training and testing sets
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train)
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)
        scores.append(fold_score)

    return scores