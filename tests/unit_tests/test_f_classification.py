

from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.statistics.f_classification import f_classification

class TestFClassification(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_f_classification(self):

        F, p = f_classification(self.dataset)

        self.assertGreater(F.shape[0], 0)
        self.assertGreater(p.shape[0], 0)

        significant_different = []
        for p_value in p:
            if p_value < 0.05:
                significant_different.append(True)
            else:
                significant_different.append(False)

        self.assertTrue(any(significant_different))
            