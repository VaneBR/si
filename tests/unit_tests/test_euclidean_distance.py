from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.statistics.euclidean_distance import euclidean_distance

class TestEuclideanDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_euclidean_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        our_distance = euclidean_distance(x, y)
        # using sklearn
        # to test this snippet, you need to install sklearn (pip install -U scikit-learn)
        from sklearn.metrics.pairwise import euclidean_distances
        sklearn_distance = euclidean_distances(x.reshape(1, -1), y)
        assert np.allclose(our_distance, sklearn_distance)