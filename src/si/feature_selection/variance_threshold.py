import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.

    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.

        Returns
        -------
        self : object
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
