import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor

class TestKNNRegressor(unittest.TestCase):
    """Tests for KNNRegressor"""
    
    def setUp(self):
        """Prepare test data"""
        self.X_train = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0]
        ])
        self.y_train = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
        
        self.train_dataset = Dataset(
            X=self.X_train, y=self.y_train,
            features=['x1', 'x2'], label='y'
        )
        
        self.X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
        self.y_test = np.array([3.0, 4.75])
        
        self.test_dataset = Dataset(
            X=self.X_test, y=self.y_test,
            features=['x1', 'x2'], label='y'
        )
    
    def test_initialization(self):
        """Test initialization"""
        knn = KNNRegressor(k=3)
        self.assertEqual(knn.k, 3)
        self.assertIsNone(knn.dataset)
    
    def test_fit_stores_dataset(self):
        """Test that fit stores dataset"""
        knn = KNNRegressor(k=3)
        knn.fit(self.train_dataset)
        
        self.assertIsNotNone(knn.dataset)
        np.testing.assert_array_equal(knn.dataset.X, self.train_dataset.X)
    
    def test_predict_returns_correct_shape(self):
        """Test predict output shape"""
        knn = KNNRegressor(k=3)
        knn.fit(self.train_dataset)
        predictions = knn.predict(self.test_dataset)
        
        self.assertEqual(predictions.shape, (len(self.X_test),))
    
    def test_predict_before_fit_raises_error(self):
        """Test predict raises error when used before fit"""
        knn = KNNRegressor(k=3)
        
        with self.assertRaises(ValueError):
            knn.predict(self.test_dataset)
    
    def test_score_returns_float(self):
        """Test score output type"""
        knn = KNNRegressor(k=3)
        knn.fit(self.train_dataset)
        score = knn.score(self.test_dataset)
        
        self.assertIsInstance(score, (float, np.floating))
        self.assertTrue(score >= 0)
    
    def test_k_equals_1(self):
        """Test behaviour with k=1"""
        knn = KNNRegressor(k=1)
        knn.fit(self.train_dataset)
        
        predictions = knn.predict(self.train_dataset)
        np.testing.assert_array_almost_equal(predictions, self.train_dataset.y)
    
    def test_different_k_values(self):
        """Test that different k values yield different outputs"""
        knn1 = KNNRegressor(k=1)
        knn3 = KNNRegressor(k=3)
        
        knn1.fit(self.train_dataset)
        knn3.fit(self.train_dataset)
        
        pred1 = knn1.predict(self.test_dataset)
        pred3 = knn3.predict(self.test_dataset)
        
        self.assertFalse(np.array_equal(pred1, pred3))

if __name__ == '__main__':
    unittest.main()