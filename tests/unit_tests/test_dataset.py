import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

class TestDatasetMethods(unittest.TestCase):
    """
    Tests for the methods dropna, fillna, and remove_by_index
    """

    def setUp(self):
        """
        Prepare the dataset for testing before each test
        """
        self.X_with_nan= np.array([[1.0, 2.0, 3.0],
                                   [4.0, np.nan, 6.0],
                                   [7.0, 8.0, 9.0],
                                   [13.0,14.0,15.0]])
        self.y = np.array(['A', 'B', 'C', 'D', 'E'])
        self.features = ['f1', 'f2', 'f3']
        self.label = 'target'
    
    def test_dropna_removes_rows_with_nan(self):
        """
        Tests if dropna correctly removes rows with NaN values
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        dataset.dropna()

        self.assertEqual(dataset.X.shape[0], 3)  # Should have 3 rows after dropping (indexes 0, 2, 4)
        self.assertEqual(len(dataset.y), 3)  # Corresponding y values should also be reduced
        self.assertFalse(np.isnan(dataset.X).any())  # No NaN values should remain
    
    def test_dropna_preserves_correct_rows(self):
        """
        Tests if dropna preserves the correct rows
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        dataset.dropna()

        expected_X = np.array([[1.0, 2.0, 3.0],
                               [7.0, 8.0, 9.0],
                               [13.0,14.0,15.0]]) #the remaining rows should be the ones from indexes 0, 2, 4
        expected_y = np.array(['A', 'C'])

        np.testing.assert_array_equal(dataset.X, expected_X)
        np.testing.assert_array_equal(dataset.y, expected_y)
    
    def test_dropna_returns_self(self):
        """
        Tests if dropna returns self 
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        result = dataset.dropna()
        self.assertIs(result, dataset)

    def test_fillna_with_numeric_value(self): 
        """
        Tests if fillna correctly fills NaN values with a numeric value
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        dataset.fillna(0)

        self.assertFalse(np.isnan(dataset.X).any())  # No NaN values should remain
        self.assertEqual(dataset.X[1, 1], 0)  # Verifies that NaN was replaced correctly
        self.assertEqual(dataset.X.shape[0], 4)  # Verifies other NaN values are also replaced correctly

    def test_fillna_with_mean(self):
        """
        Tests if fillna correctly fills NaN values with the mean of the column
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        dataset.fillna("mean")

        self.assertFalse(np.isnan(dataset.X).any())  # No NaN values should remain
        self.assertAlmostEqual(dataset.X[3, 0], 6.25) # Verifies that NaN was replaced with mean correctly (1 + 4 + 7 + 13) / 4 = 6.25
        self.assertAlmostEqual(dataset.X[1, 1], 8.75)  # Verifies that NaN was replaced with mean correctly (2 + 8 + 11 + 14) / 4 = 8.75

    def test_fillna_with_median(self):
        """
        Tests if fillna correctly fills NaN values with the median of the column
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        dataset.fillna("median")

        self.assertFalse(np.isnan(dataset.X).any())  # No NaN values should remain
        self.assertAlmostEqual(dataset.X[3, 0], 5.5)  # Verifies that NaN was replaced with median correctly [1, 4, 7, 13] -> 5.5
        self.assertAlmostEqual(dataset.X[1, 1], 9.5)  # Verifies that NaN was replaced with median correctly [2, 8, 11, 14] -> 9.5

    def test_fillna_returns_self(self):
        """
        Tests if fillna returns self 
        """
        dataset= Dataset(self.X_with_nan.copy(), self.y.copy(),
                         self.features, self.label)
        result = dataset.fillna(0)
        self.assertIs(result, dataset)

    def test_fillna_invalid_value_raises_error(self):
        """
        Tests if fillna raises ValueError for invalid method
        """
        dataset = Dataset(self.X_with_nan.copy(), self.y.copy(), 
                         self.features, self.label)
        
        with self.assertRaises(ValueError):
            dataset.fillna("invalid")
    
    def test_remove_by_index_removes_correct_row(self):
        """
        Tests if remove_by_index correctly removes the specified row
        """
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array(['A', 'B', 'C', 'D'])
        dataset = Dataset(X.copy(), y.copy(), ['f1', 'f2'], 'target')

        dataset.remove_by_index(1)
        
        expected_X = np.array([[1, 2], [5, 6], [7, 8]])
        expected_y = np.array(['A', 'C', 'D'])
        
        np.testing.assert_array_equal(dataset.X, expected_X)
        np.testing.assert_array_equal(dataset.y, expected_y)

    def test_remove_by_index_first_row(self):
        """
        Tests if remove_by_index correctly removes the first row
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array(['A', 'B', 'C'])
        dataset = Dataset(X.copy(), y.copy(), ['f1', 'f2'], 'target')

        dataset.remove_by_index(0)
        
        expected_X = np.array([[3, 4], [5, 6]])
        expected_y = np.array(['B', 'C'])
        
        np.testing.assert_array_equal(dataset.X, expected_X)
        np.testing.assert_array_equal(dataset.y, expected_y)
    
    def test_remove_by_index_last_row(self):
        """
        Tests if remove_by_index correctly removes the last row
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array(['A', 'B', 'C'])
        dataset = Dataset(X.copy(), y.copy(), ['f1', 'f2'], 'target')

        dataset.remove_by_index(2)
        
        expected_X = np.array([[1, 2], [3, 4]])
        expected_y = np.array(['A', 'B'])
        
        np.testing.assert_array_equal(dataset.X, expected_X)
        np.testing.assert_array_equal(dataset.y, expected_y)
    
    def test_remove_by_index_returns_self(self):
        """
        Tests if remove_by_index returns self 
        """
        X = np.array([[1, 2], [3, 4]])
        y = np.array(['A', 'B'])
        dataset = Dataset(X, y, ['f1', 'f2'], 'target')

        result = dataset.remove_by_index(0)
        self.assertIs(result, dataset)
    
    def test_remove_by_index_invalid_index_raises_error(self):
        """
        Tests if remove_by_index raises IndexError for invalid index
        """
        X = np.array([[1, 2], [3, 4]])
        y = np.array(['A', 'B'])
        dataset = Dataset(X, y, ['f1', 'f2'], 'target')

        with self.assertRaises(IndexError):
            dataset.remove_by_index(5)
        
        with self.assertRaises(IndexError):
            dataset.remove_by_index(-1)

if __name__ == '__main__':
    unittest.main()