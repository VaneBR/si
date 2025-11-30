import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.random_forest_classifier import RandomForestClassifier
from si.models.decision_tree_classifier import DecisionTreeClassifier


class TestRandomForestClassifier(unittest.TestCase):
    """Tests for the RandomForestClassifier class"""
    
    def setUp(self):
        """Prepare test data"""
        np.random.seed(42)
        
        # Synthetic dataset for classification
        n_samples = 100
        n_features = 4
        
        self.X = np.random.randn(n_samples, n_features)
        
        # 3 classes based on simple rules
        self.y = np.array(['A'] * 30 + ['B'] * 40 + ['C'] * 30)
        
        # Add some structure to the data
        self.X[:30, 0] += 2.0   # Class A: feature 0 higher
        self.X[30:70, 1] += 2.0  # Class B: feature 1 higher
        self.X[70:, 2] += 2.0    # Class C: feature 2 higher
        
        self.dataset = Dataset(
            X=self.X, 
            y=self.y,
            features=['f1', 'f2', 'f3', 'f4'],
            label='target'
        )
        
        # Split into train and test
        split_idx = 80
        self.train = Dataset(
            X=self.X[:split_idx],
            y=self.y[:split_idx],
            features=['f1', 'f2', 'f3', 'f4'],
            label='target'
        )
        self.test = Dataset(
            X=self.X[split_idx:],
            y=self.y[split_idx:],
            features=['f1', 'f2', 'f3', 'f4'],
            label='target'
        )
    
    def test_initialization(self):
        """Tests initialization"""
        rf = RandomForestClassifier(
            n_estimators=10,
            max_features=2,
            min_sample_split=2,
            max_depth=5,
            mode='gini',
            seed=42
        )
        
        self.assertEqual(rf.n_estimators, 10)
        self.assertEqual(rf.max_features, 2)
        self.assertEqual(rf.min_sample_split, 2)
        self.assertEqual(rf.max_depth, 5)
        self.assertEqual(rf.mode, 'gini')
        self.assertEqual(rf.seed, 42)
        self.assertEqual(len(rf.trees), 0)
    
    def test_fit_creates_trees(self):
        """Tests whether fit creates trees"""
        rf = RandomForestClassifier(n_estimators=5, seed=42)
        rf.fit(self.train)
        
        self.assertEqual(len(rf.trees), 5)
        
        # Each element must be a tuple (features, tree)
        for features, tree in rf.trees:
            self.assertIsInstance(features, np.ndarray)
            self.assertIsInstance(tree, DecisionTreeClassifier)
    
    def test_fit_returns_self(self):
        """Tests whether fit returns self"""
        rf = RandomForestClassifier(n_estimators=5, seed=42)
        result = rf.fit(self.train)
        
        self.assertIs(result, rf)
    
    def test_max_features_auto(self):
        """Tests whether max_features None uses sqrt(n_features)"""
        rf = RandomForestClassifier(n_estimators=5, max_features=None, seed=42)
        rf.fit(self.train)
        
        # sqrt(4) = 2
        expected_max_features = int(np.sqrt(4))
        self.assertEqual(rf.max_features, expected_max_features)
    
    def test_max_features_not_exceed_n_features(self):
        """Tests that max_features does not exceed n_features"""
        rf = RandomForestClassifier(n_estimators=5, max_features=100, seed=42)
        rf.fit(self.train)
        
        # Should be limited to number of features
        self.assertEqual(rf.max_features, 4)
    
    def test_predict_returns_correct_shape(self):
        """Tests whether predict returns the correct shape"""
        rf = RandomForestClassifier(n_estimators=5, seed=42)
        rf.fit(self.train)
        
        predictions = rf.predict(self.test)
        
        self.assertEqual(predictions.shape, (len(self.test.y),))
    
    def test_predict_returns_valid_classes(self):
        """Tests whether predict returns only valid classes"""
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(self.train)
        
        predictions = rf.predict(self.test)
        
        valid_classes = set(self.train.y)
        for pred in predictions:
            self.assertIn(pred, valid_classes)
    
    def test_predict_before_fit_raises_error(self):
        """Tests whether predict raises error if called before fit"""
        rf = RandomForestClassifier(n_estimators=5, seed=42)
        
        with self.assertRaises(ValueError):
            rf.predict(self.test)
    
    def test_score_returns_float(self):
        """Tests whether score returns a float between 0 and 1"""
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(self.train)
        
        score = rf.score(self.test)
        
        self.assertIsInstance(score, (float, np.floating))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_reproducibility_with_seed(self):
        """Tests reproducibility with seed"""
        rf1 = RandomForestClassifier(n_estimators=10, seed=42)
        rf1.fit(self.train)
        pred1 = rf1.predict(self.test)
        
        rf2 = RandomForestClassifier(n_estimators=10, seed=42)
        rf2.fit(self.train)
        pred2 = rf2.predict(self.test)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_different_seeds_give_different_results(self):
        """Tests that different seeds usually give different results"""
        rf1 = RandomForestClassifier(n_estimators=5, seed=42)
        rf1.fit(self.train)
        pred1 = rf1.predict(self.test)
        
        rf2 = RandomForestClassifier(n_estimators=5, seed=123)
        rf2.fit(self.train)
        pred2 = rf2.predict(self.test)
        
        # Trees should differ (highly likely)
        different_trees = False
        for (f1, t1), (f2, t2) in zip(rf1.trees, rf2.trees):
            if not np.array_equal(f1, f2):
                different_trees = True
                break
        
        self.assertTrue(different_trees)
    
    def test_more_estimators_generally_better(self):
        """Tests that more estimators generally improve performance"""
        rf_small = RandomForestClassifier(n_estimators=1, seed=42)
        rf_small.fit(self.train)
        score_small = rf_small.score(self.test)
        
        rf_large = RandomForestClassifier(n_estimators=50, seed=42)
        rf_large.fit(self.train)
        score_large = rf_large.score(self.test)
        
        self.assertGreaterEqual(score_large, score_small - 0.1)
    
    def test_gini_and_entropy_both_work(self):
        """Tests that both gini and entropy criteria work"""
        for mode in ['gini', 'entropy']:
            rf = RandomForestClassifier(n_estimators=5, mode=mode, seed=42)
            rf.fit(self.train)
            score = rf.score(self.test)
            
            # Should achieve reasonable accuracy
            self.assertGreater(score, 0.3)
    
    def test_get_feature_importance(self):
        """Tests the get_feature_importance method"""
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(self.train)
        
        importance = rf.get_feature_importance(self.dataset)
        
        # Must return a dictionary
        self.assertIsInstance(importance, dict)
        
        # Must contain one entry per feature
        self.assertEqual(len(importance), 4)
        
        # All importances must lie between 0 and 1
        for feature, imp in importance.items():
            self.assertGreaterEqual(imp, 0.0)
            self.assertLessEqual(imp, 1.0)
    
    def test_get_feature_importance_before_fit_raises_error(self):
        """Tests whether get_feature_importance raises error before fit"""
        rf = RandomForestClassifier(n_estimators=5)
        
        with self.assertRaises(ValueError):
            rf.get_feature_importance(self.dataset)
    
    def test_bootstrap_sampling(self):
        """Tests whether bootstrap sampling is working"""
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(self.train)
        
        # Trees should use different feature subsets
        feature_sets = [set(features) for features, tree in rf.trees]
        
        all_same = all(fs == feature_sets[0] for fs in feature_sets)
        self.assertFalse(all_same)
    
    def test_max_depth_affects_performance(self):
        """Tests that max_depth affects performance"""
        rf_shallow = RandomForestClassifier(
            n_estimators=10, 
            max_depth=2, 
            seed=42
        )
        rf_shallow.fit(self.train)
        score_shallow = rf_shallow.score(self.train)
        
        rf_deep = RandomForestClassifier(
            n_estimators=10, 
            max_depth=20, 
            seed=42
        )
        rf_deep.fit(self.train)
        score_deep = rf_deep.score(self.train)
        
        # Deep trees should score better on training
        self.assertGreater(score_deep, score_shallow)
    
    def test_min_sample_split_affects_trees(self):
        """Tests that min_sample_split affects trees"""
        rf1 = RandomForestClassifier(
            n_estimators=5,
            min_sample_split=2,
            seed=42
        )
        rf1.fit(self.train)
        
        rf2 = RandomForestClassifier(
            n_estimators=5,
            min_sample_split=20,
            seed=42
        )
        rf2.fit(self.train)
        
        score1 = rf1.score(self.train)
        score2 = rf2.score(self.train)
        
        self.assertGreaterEqual(score1, score2)
    
    def test_voting_mechanism(self):
        """Tests the voting mechanism"""
        rf = RandomForestClassifier(n_estimators=3, seed=42)
        rf.fit(self.train)
        
        predictions = rf.predict(self.test[:1])
        
        self.assertIn(predictions[0], set(self.train.y))
    
    def test_performance_better_than_single_tree(self):
        """Tests that RF generally performs better than a single tree"""
        dt = DecisionTreeClassifier(
            min_samples_split=2,
            max_depth=10,
            mode='gini'
        )
        dt.fit(self.train)
        dt_score = dt.score(self.test)
        
        rf = RandomForestClassifier(
            n_estimators=20,
            max_depth=10,
            mode='gini',
            seed=42
        )
        rf.fit(self.train)
        rf_score = rf.score(self.test)
        
        self.assertGreaterEqual(rf_score, dt_score - 0.15)

if __name__ == '__main__':
    unittest.main()
