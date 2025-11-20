from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.feature_selection.select_k_best import SelectKBest
from si.io.csv_file import read_csv

from si.statistics.f_classification import f_classification

class TestSelectKBest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):

        select_k_best = SelectKBest(score_func = f_classification, k = 2)

        select_k_best.fit(self.dataset)
        self.assertTrue(select_k_best.F.shape[0] > 0)
        self.assertTrue(select_k_best.p.shape[0] > 0)

    def test_transform(self):
        select_k_best = SelectKBest(score_func = f_classification, k = 2)

        select_k_best.fit(self.dataset)
        new_dataset = select_k_best.transform(self.dataset)

        self.assertLess(len(new_dataset.features), len(self.dataset.features))
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])