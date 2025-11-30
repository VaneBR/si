import unittest
import numpy as np
from si.data.dataset import Dataset
from si.metrics.rmse import rmse

class TestRMSE(unittest.TestCase):
    """Tests for RMSE function"""
    
    def test_perfect_prediction(self):
        """Test RMSE with perfect prediction (should be 0)"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        error = rmse(y_true, y_pred)
        self.assertAlmostEqual(error, 0.0)
    
    def test_known_rmse(self):
        """Test RMSE with known values"""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        
        differences = y_true - y_pred
        squared = differences ** 2
        mse = np.mean(squared)
        expected_rmse = np.sqrt(mse)
        
        error = rmse(y_true, y_pred)
        self.assertAlmostEqual(error, expected_rmse, places=3)
    
    def test_size_mismatch_raises_error(self):
        """Test error when array sizes differ"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        
        with self.assertRaises(ValueError):
            rmse(y_true, y_pred)
    
    def test_empty_arrays_raise_error(self):
        """Test error when arrays are empty"""
        y_true = np.array([])
        y_pred = np.array([])
        
        with self.assertRaises(ValueError):
            rmse(y_true, y_pred)

if __name__ == '__main__':
    unittest.main(verbosity=2)