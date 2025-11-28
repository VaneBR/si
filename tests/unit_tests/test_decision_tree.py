from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier

class TestDecisionTree(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.train_dataset)

        self.assertEqual(decision_tree.min_sample_split, 2)
        self.assertEqual(decision_tree.max_depth, 10)


    def test_predict(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.train_dataset)

        predictions = decision_tree.predict(self.test_dataset)
        print(predictions)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.train_dataset)
        accuracy_ = decision_tree.score(self.test_dataset)

        self.assertEqual(round(accuracy_, 2), 0.92)