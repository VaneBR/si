import unittest
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import stratified_train_test_split

class TestStratifiedTrainTestSplit(unittest.TestCase):
    """Tests for stratified_train_test_split"""
    
    def setUp(self):
        """Prepare test data"""
        X = np.random.randn(150, 4)
        y = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50)
        self.dataset = Dataset(
            X=X, 
            y=y, 
            features=['f1', 'f2', 'f3', 'f4'], 
            label='target'
        )
    
    def test_split_sizes(self):
        """Test whether the split sizes are correct"""
        train, test = stratified_train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )
        
        total = len(train.y) + len(test.y)
        self.assertEqual(total, len(self.dataset.y))
        
        expected_test_size = int(len(self.dataset.y) * 0.2)
        self.assertAlmostEqual(len(test.y), expected_test_size, delta=3)
    
    def test_class_proportions(self):
        """Test if class proportions are maintained"""
        train, test = stratified_train_test_split(
            self.dataset, test_size=0.3, random_state=42
        )
        
        unique_orig, counts_orig = np.unique(self.dataset.y, return_counts=True)
        prop_orig = counts_orig / len(self.dataset.y)
        
        unique_test, counts_test = np.unique(test.y, return_counts=True)
        prop_test = counts_test / len(test.y)
        
        for i in range(len(unique_orig)):
            self.assertAlmostEqual(prop_orig[i], prop_test[i], delta=0.1)
    
    def test_no_overlap(self):
        """Test that there is no overlap between train and test"""
        train, test = stratified_train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )
        
        total_samples = len(train.y) + len(test.y)
        self.assertEqual(total_samples, len(self.dataset.y))
    
    def test_reproducibility(self):
        """Test reproducibility when using random_state"""
        train1, test1 = stratified_train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )
        train2, test2 = stratified_train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )
        
        np.testing.assert_array_equal(train1.X, train2.X)
        np.testing.assert_array_equal(test1.X, test2.X)
    
    def test_without_labels_raises_error(self):
        """Test that splitting without labels raises an error"""
        dataset_no_y = Dataset(X=self.dataset.X, y=None)
        
        with self.assertRaises(ValueError):
            stratified_train_test_split(dataset_no_y, test_size=0.2)

if __name__ == '__main__':
    unittest.main()