import unittest
import numpy as np
from si.data.dataset import Dataset
from si.decomposition.pca import PCA

class TestPCA(unittest.TestCase):
    """Tests for PCA class"""
    
    def setUp(self):
        """Prepare test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4)
        self.y = np.random.choice(['A', 'B'], size=100)
        self.dataset = Dataset(
            X=self.X, 
            y=self.y, 
            features=['f1', 'f2', 'f3', 'f4'], 
            label='target'
        )
    
    def test_initialization(self):
        """Test initialization"""
        pca = PCA(n_components=2)
        self.assertEqual(pca.n_components, 2)
        self.assertIsNone(pca.mean)
        self.assertIsNone(pca.components)
        self.assertIsNone(pca.explained_variance)
    
    def test_fit_calculates_parameters(self):
        """Test whether fit computes parameters"""
        pca = PCA(n_components=2)
        pca.fit(self.dataset)
        
        self.assertIsNotNone(pca.mean)
        self.assertIsNotNone(pca.components)
        self.assertIsNotNone(pca.explained_variance)
        
        # Check shapes
        self.assertEqual(pca.mean.shape, (4,))
        self.assertEqual(pca.components.shape, (2, 4))
        self.assertEqual(pca.explained_variance.shape, (2,))
    
    def test_transform_reduces_dimensions(self):
        """Test whether transform reduces dimensionality"""
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(self.dataset)
        
        self.assertEqual(transformed.X.shape[1], 2)
        self.assertEqual(transformed.X.shape[0], self.dataset.X.shape[0])
    
    def test_explained_variance_sum(self):
        """Test whether explained variance values are valid"""
        pca = PCA(n_components=2)
        pca.fit(self.dataset)
        
        self.assertTrue(np.all(pca.explained_variance >= 0))
        self.assertTrue(np.all(pca.explained_variance <= 1))
        
        self.assertTrue(np.sum(pca.explained_variance) <= 1)
    
    def test_transform_before_fit_raises_error(self):
        """Test if transform raises error when called before fit"""
        pca = PCA(n_components=2)
        
        with self.assertRaises(ValueError):
            pca.transform(self.dataset)
    
    def test_features_named_correctly(self):
        """Test whether transformed feature names are correct"""
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(self.dataset)
        
        self.assertEqual(transformed.features, ['PC1', 'PC2', 'PC3'])

if __name__ == '__main__':
    unittest.main()
