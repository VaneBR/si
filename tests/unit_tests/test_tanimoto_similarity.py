import unittest
import numpy as np
from si.data.dataset import Dataset
from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(unittest.TestCase):
    """Tests for tanimoto_similarity function"""
    
    def test_identical_samples(self):
        """Test distance between identical samples (should be 0)"""
        x = np.array([1, 0, 1, 1, 0])
        y = np.array([[1, 0, 1, 1, 0]])
        distance = tanimoto_similarity(x, y)
        self.assertAlmostEqual(distance[0], 0.0)
    
    def test_completely_different(self):
        """Test distance between completely different samples (should be 1)"""
        x = np.array([1, 1, 1, 0, 0])
        y = np.array([[0, 0, 0, 1, 1]])
        distance = tanimoto_similarity(x, y)
        self.assertAlmostEqual(distance[0], 1.0)
    
    def test_multiple_samples(self):
        """Test calculation with multiple samples"""
        x = np.array([1, 0, 1, 1])
        y = np.array([
            [1, 0, 1, 1],  # identical
            [1, 1, 0, 0],  # partially different
            [0, 0, 0, 0]   # no overlap
        ])
        distances = tanimoto_similarity(x, y)
        
        self.assertEqual(len(distances), 3)
        self.assertAlmostEqual(distances[0], 0.0)   # identical
        self.assertTrue(0 < distances[1] < 1)       # partial
        self.assertAlmostEqual(distances[2], 1.0)   # different
    
    def test_input_validation(self):
        """Test input validation"""
        x = np.array([1, 0, 1])
        
        # x must be 1D
        with self.assertRaises(ValueError):
            tanimoto_similarity(np.array([[1, 0, 1]]), np.array([[1, 0, 1]]))
        
        # y must be 2D
        with self.assertRaises(ValueError):
            tanimoto_similarity(x, np.array([1, 0, 1]))
        
        # Incompatible dimensions
        with self.assertRaises(ValueError):
            tanimoto_similarity(x, np.array([[1, 0]]))


if __name__ == '__main__':
    unittest.main()
