from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.layers import DenseLayer
from si.neural_networks.optimizers import Optimizer

class MockOptimizer(Optimizer):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        return w - self.learning_rate * grad_loss_w

class TestDenseLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_forward_propagation(self):

        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(0.001))
        output = dense_layer.forward_propagation(self.dataset.X, training=False)
        self.assertEqual(output.shape[0], self.dataset.X.shape[0])
        self.assertEqual(output.shape[1], 30)


    def test_backward_propagation(self):
        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(learning_rate=0.001))
        dense_layer.forward_propagation(self.dataset.X, training=True)
        input_error = dense_layer.backward_propagation(output_error=np.random.random((self.dataset.X.shape[0], 30)))
        self.assertEqual(input_error.shape[0], self.dataset.X.shape[0])
        self.assertEqual(input_error.shape[1], 9)