import os
from unittest import TestCase

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.feature_selection.variance_threshold import VarianceThreshold


class TestVarianceThreshold(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):

        estimator = VarianceThreshold(threshold=0.6).fit(self.dataset)

        self.assertEqual(estimator.threshold, 0.6)
        self.assertEqual(estimator.variance.shape[0], 4)

    def test_transform(self):

        estimator = VarianceThreshold(threshold=0.6).fit(self.dataset)
        new_dataset = estimator.transform(self.dataset)

        self.assertGreater(len(self.dataset.features), len(new_dataset.features))
        self.assertGreater(self.dataset.X.shape[1], new_dataset.X.shape[1])