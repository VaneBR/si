import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the euclidean distance of a point (x) to a set of points y.
        distance_x_y1 = sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
        distance_x_y2 = sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Euclidean distance for each point in y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))


if __name__ == '__main__':
    # test euclidean_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = euclidean_distance(x, y)
    # using sklearn
    # to test this snippet, you need to install sklearn (pip install -U scikit-learn)
    from sklearn.metrics.pairwise import euclidean_distances
    sklearn_distance = euclidean_distances(x.reshape(1, -1), y)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance)