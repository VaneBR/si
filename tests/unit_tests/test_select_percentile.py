import unittest
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(unittest.TestCase):
    """
    Tests for the SelectPercentile class
    """
    
    def setUp(self):
        """
        Prepare test data
        """
        # Simple dataset for classification
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.choice(['A', 'B', 'C'], size=100)
        self.features = [f'feature_{i}' for i in range(10)]
        self.dataset = Dataset(X=self.X, y=self.y, 
                               features=self.features, label='target')
    
    def test_initialization(self):
        """
        Tests correct initialization
        """
        selector = SelectPercentile(percentile=50)
        
        self.assertEqual(selector.percentile, 50)
        self.assertIsNone(selector.F)
        self.assertIsNone(selector.p)
    
    def test_invalid_percentile_raises_error(self):
        """
        Tests whether an invalid percentile raises an error
        """
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-10)
        
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=150)
    
    def test_fit_calculates_f_and_p(self):
        """
        Tests whether fit computes F and p values
        """
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)
        
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        self.assertEqual(len(selector.F), 10)
        self.assertEqual(len(selector.p), 10)
    
    def test_fit_returns_self(self):
        """
        Tests whether fit returns self
        """
        selector = SelectPercentile(percentile=50)
        result = selector.fit(self.dataset)
        
        self.assertIs(result, selector)
    
    def test_transform_before_fit_raises_error(self):
        """
        Tests whether transform without fit raises an error
        """
        selector = SelectPercentile(percentile=50)
        
        with self.assertRaises(ValueError):
            selector.transform(self.dataset)
    
    def test_transform_selects_correct_number_of_features(self):
        """
        Tests whether transform selects the correct number of features
        """
        # Test with 50%
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)
        transformed = selector.transform(self.dataset)
        
        # 50% of 10 features = 5 features
        self.assertEqual(transformed.X.shape[1], 5)
        
        # Test with 30%
        selector = SelectPercentile(percentile=30)
        selector.fit(self.dataset)
        transformed = selector.transform(self.dataset)
        
        # 30% of 10 features = 3 features
        self.assertEqual(transformed.X.shape[1], 3)
    
    def test_transform_preserves_number_of_samples(self):
        """
        Tests whether transform preserves the number of samples
        """
        selector = SelectPercentile(percentile=50)
        transformed = selector.fit_transform(self.dataset)
        
        self.assertEqual(transformed.X.shape[0], self.dataset.X.shape[0])
    
    def test_transform_preserves_y(self):
        """
        Tests whether transform preserves y
        """
        selector = SelectPercentile(percentile=50)
        transformed = selector.fit_transform(self.dataset)
        
        np.testing.assert_array_equal(transformed.y, self.dataset.y)
    
    def test_transform_updates_features(self):
        """
        Tests whether transform updates the feature names list
        """
        selector = SelectPercentile(percentile=50)
        transformed = selector.fit_transform(self.dataset)
        
        self.assertIsNotNone(transformed.features)
        self.assertEqual(len(transformed.features), 5)
        
        # Check that selected features are a subset of the original list
        for feature in transformed.features:
            self.assertIn(feature, self.features)
    
    def test_fit_transform(self):
        """
        Tests the fit_transform method
        """
        selector = SelectPercentile(percentile=40)
        transformed = selector.fit_transform(self.dataset)
        
        # 40% of 10 = 4 features
        self.assertEqual(transformed.X.shape[1], 4)
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
    
    def test_extreme_percentiles(self):
        """
        Tests extreme percentiles
        """
        # 10% - must select at least 1 feature
        selector = SelectPercentile(percentile=10)
        transformed = selector.fit_transform(self.dataset)
        self.assertEqual(transformed.X.shape[1], 1)
        
        # 100% - must select all features
        selector = SelectPercentile(percentile=100)
        transformed = selector.fit_transform(self.dataset)
        self.assertEqual(transformed.X.shape[1], 10)
    
    def test_example_from_documentation(self):
        """
        Tests the specific example from documentation:
        10 features, F-values [1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2]
        percentile=40 should select [5.6, 5.6, 7.8, 6.5]
        """
        # Create dataset with known F-values
        X = np.random.randn(50, 10)
        
        # Creating y that produces the exact desired F-values is complex,
        # so we test the selection logic manually.
        selector = SelectPercentile(percentile=40)
        selector.F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        selector.p = np.array([0.1] * 10)  # p-values don't matter here
        
        # Dummy dataset
        y = np.random.choice(['A', 'B'], size=50)
        dataset = Dataset(X=X, y=y)
        
        # Transform
        transformed = selector.transform(dataset)
        
        # Must select 4 features (40% of 10)
        self.assertEqual(transformed.X.shape[1], 4)
        
        # Check selected F-values
        mask = selector._get_feature_mask()
        selected_f_values = selector.F[mask]
        
        # Must select exactly 4
        self.assertEqual(len(selected_f_values), 4)
        # All must be >= 5.6
        self.assertTrue(all(f >= 5.6 for f in selected_f_values))
        # Must include 7.8 and 6.5
        self.assertIn(7.8, selected_f_values)
        self.assertIn(6.5, selected_f_values)
    
    def test_consistent_results(self):
        """
        Tests whether multiple calls return consistent results
        """
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)
        
        transformed1 = selector.transform(self.dataset)
        transformed2 = selector.transform(self.dataset)
        
        np.testing.assert_array_equal(transformed1.X, transformed2.X)
        self.assertEqual(transformed1.features, transformed2.features)


if __name__ == '__main__':
    unittest.main()
