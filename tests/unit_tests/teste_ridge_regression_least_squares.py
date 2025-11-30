import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

class TestRidgeRegressionLeastSquares(unittest.TestCase):
    """Tests for the RidgeRegressionLeastSquares class"""
    
    def setUp(self):
        """Prepare test data"""
        np.random.seed(42)
        
        # Synthetic dataset: y = 3 + 2*x1 + 5*x2 + noise
        n_samples = 100
        self.X = np.random.randn(n_samples, 2)
        self.y = 3 + 2*self.X[:, 0] + 5*self.X[:, 1] + np.random.randn(n_samples) * 0.1
        
        self.dataset = Dataset(
            X=self.X, 
            y=self.y, 
            features=['x1', 'x2'], 
            label='y'
        )
        
        # Train-test split
        split_idx = int(0.8 * n_samples)
        self.train_dataset = Dataset(
            X=self.X[:split_idx], 
            y=self.y[:split_idx],
            features=['x1', 'x2'], 
            label='y'
        )
        self.test_dataset = Dataset(
            X=self.X[split_idx:], 
            y=self.y[split_idx:],
            features=['x1', 'x2'], 
            label='y'
        )
    
    def test_initialization(self):
        """Tests initialization"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        
        self.assertEqual(ridge.l2_penalty, 1.0)
        self.assertTrue(ridge.scale)
        self.assertIsNone(ridge.theta)
        self.assertIsNone(ridge.theta_zero)
    
    def test_fit_estimates_parameters(self):
        """Tests whether fit estimates parameters"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)
        
        self.assertIsNotNone(ridge.theta)
        self.assertIsNotNone(ridge.theta_zero)
        self.assertIsNotNone(ridge.mean)
        self.assertIsNotNone(ridge.std)
        
        # Check shapes
        self.assertEqual(ridge.theta.shape, (2,))
        self.assertEqual(ridge.mean.shape, (2,))
        self.assertEqual(ridge.std.shape, (2,))
    
    def test_fit_returns_self(self):
        """Tests whether fit returns self"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        result = ridge.fit(self.train_dataset)
        
        self.assertIs(result, ridge)
    
    def test_predict_returns_correct_shape(self):
        """Tests whether predict returns correct output shape"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)
        
        predictions = ridge.predict(self.test_dataset)
        
        self.assertEqual(predictions.shape, (len(self.test_dataset.y),))
    
    def test_predict_before_fit_raises_error(self):
        """Tests whether predict raises error before fitting"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        
        with self.assertRaises(ValueError):
            ridge.predict(self.test_dataset)
    
    def test_score_returns_float(self):
        """Tests whether score returns a float"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)
        
        score = ridge.score(self.test_dataset)
        
        self.assertIsInstance(score, (float, np.floating))
        self.assertTrue(score >= 0)
    
    def test_no_regularization_recovers_coefficients(self):
        """Tests whether no regularization recovers true coefficients"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=0.0, scale=True)
        ridge.fit(self.train_dataset)
        
        # For clean synthetic data, coefficients should be close to [2, 5]
        # and intercept close to 3
        self.assertAlmostEqual(ridge.theta_zero, 3.0, delta=0.5)
        self.assertAlmostEqual(ridge.theta[0], 2.0, delta=0.5)
        self.assertAlmostEqual(ridge.theta[1], 5.0, delta=0.5)
    
    def test_regularization_reduces_coefficient_magnitude(self):
        """Tests whether regularization reduces coefficient magnitude"""
        ridge_no_reg = RidgeRegressionLeastSquares(l2_penalty=0.0, scale=True)
        ridge_no_reg.fit(self.train_dataset)
        
        ridge_with_reg = RidgeRegressionLeastSquares(l2_penalty=10.0, scale=True)
        ridge_with_reg.fit(self.train_dataset)
        
        # Norm of coefficients should be smaller with regularization
        norm_no_reg = np.linalg.norm(ridge_no_reg.theta)
        norm_with_reg = np.linalg.norm(ridge_with_reg.theta)
        
        self.assertLess(norm_with_reg, norm_no_reg)
    
    def test_increasing_regularization_decreases_norm(self):
        """Tests whether increasing regularization decreases coefficient norm"""
        l2_values = [0.1, 1.0, 10.0, 100.0]
        norms = []
        
        for l2 in l2_values:
            ridge = RidgeRegressionLeastSquares(l2_penalty=l2, scale=True)
            ridge.fit(self.train_dataset)
            norms.append(np.linalg.norm(ridge.theta))
        
        # Norms should be decreasing
        for i in range(len(norms) - 1):
            self.assertGreater(norms[i], norms[i+1])
    
    def test_scale_true_vs_false(self):
        """Tests difference between scaled and unscaled regression"""
        ridge_scaled = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge_scaled.fit(self.train_dataset)
        
        ridge_no_scale = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=False)
        ridge_no_scale.fit(self.train_dataset)
        
        # Coefficients should differ
        self.assertFalse(np.allclose(ridge_scaled.theta, ridge_no_scale.theta))
        
        # Predictions should be similar
        pred_scaled = ridge_scaled.predict(self.test_dataset)
        pred_no_scale = ridge_no_scale.predict(self.test_dataset)
        
        # MSE should be similar (within reasonable margin)
        mse_scaled = np.mean((self.test_dataset.y - pred_scaled)**2)
        mse_no_scale = np.mean((self.test_dataset.y - pred_no_scale)**2)
        
        self.assertTrue(abs(mse_scaled - mse_no_scale) < 1.0)
    
    def test_get_coefficients_returns_dict(self):
        """Tests whether get_coefficients returns correct dictionary"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)
        
        coeffs = ridge.get_coefficients()
        
        self.assertIsInstance(coeffs, dict)
        self.assertIn('intercept', coeffs)
        self.assertIn('coefficients', coeffs)
        self.assertEqual(len(coeffs['coefficients']), 2)
    
    def test_get_coefficients_before_fit_raises_error(self):
        """Tests whether get_coefficients raises error before fit"""
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        
        with self.assertRaises(ValueError):
            ridge.get_coefficients()
    
    def test_intercept_not_regularized(self):
        """Tests whether the intercept is not regularized"""
        mean_y = np.mean(self.train_dataset.y)
        
        ridge_low = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        ridge_low.fit(self.train_dataset)
        
        ridge_high = RidgeRegressionLeastSquares(l2_penalty=100.0, scale=True)
        ridge_high.fit(self.train_dataset)
        
        # Intercepts should be similar
        self.assertAlmostEqual(ridge_low.theta_zero, ridge_high.theta_zero, delta=0.5)
        
        # Both should be near the mean
        self.assertAlmostEqual(ridge_low.theta_zero, mean_y, delta=1.0)
    
    def test_consistency_multiple_fits(self):
        """Tests consistency across multiple fits"""
        ridge1 = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge1.fit(self.train_dataset)
        pred1 = ridge1.predict(self.test_dataset)
        
        ridge2 = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge2.fit(self.train_dataset)
        pred2 = ridge2.predict(self.test_dataset)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)

if __name__ == '__main__':
    unittest.main()