import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy
from typing import Callable
from collections import Counter

# models/random_forest_classifier.py

class RandomForestClassifier(Model):
    """
    Random Forest Classifier - Ensemble of Decision Trees.
    
    Random Forest is an ensemble technique that combines multiple decision trees
    to improve prediction accuracy and reduce overfitting.
    Each tree is trained on a bootstrap sample (sampling with replacement)
    and uses only a random subset of features.
    
    Parameters
    ----------
    n_estimators : int
        Number of decision trees to use (default: 100)
    max_features : int, optional
        Maximum number of features per tree.
        If None, uses sqrt(n_features) (default: None)
    min_sample_split : int
        Minimum number of samples required to split (default: 2)
    max_depth : int
        Maximum depth of trees (default: 10)
    mode : str
        Impurity metric: 'gini' or 'entropy' (default: 'gini')
    seed : int, optional
        Seed for reproducibility (default: None)
    
    Attributes
    ----------
    trees : list
        List of tuples (used_features, trained_tree)
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_features: int = None,
                 min_sample_split: int = 2,
                 max_depth: int = 10,
                 mode: str = 'gini',
                 seed: int = None):
        """
        Initializes the RandomForestClassifier.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_features : int, optional
            Maximum number of features per tree (None = sqrt(n_features))
        min_sample_split : int
            Minimum samples required for split
        max_depth : int
            Maximum tree depth
        mode : str
            'gini' or 'entropy'
        seed : int, optional
            Random seed
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        
        # Estimated parameters
        self.trees = []  # List of (features, tree)
    
    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Trains the decision trees of the Random Forest.
        
        Steps:
        1. Set random seed
        2. Determine max_features if None (sqrt(n_features))
        3. For each tree:
           a. Create bootstrap dataset (with replacement)
           b. Select random subset of features
           c. Train decision tree
           d. Store (features, tree)
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset
        
        Returns
        -------
        self : RandomForestClassifier
            Returns self
        """
        if dataset.y is None:
            raise ValueError("Dataset must contain labels (y) for classification")
        
        # 1. Set random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
        
        n_samples = dataset.X.shape[0]
        n_features = dataset.X.shape[1]
        
        # 2. Determine max_features if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        # Ensure max_features does not exceed total features
        self.max_features = min(self.max_features, n_features)
        
        # Clear previous trees (if fit is called multiple times)
        self.trees = []
        
        # Create and train n_estimators trees
        for i in range(self.n_estimators):
            # 3. Bootstrap dataset
            
            # a) Select n_samples with replacement
            bootstrap_indices = np.random.choice(n_samples, 
                                                size=n_samples, 
                                                replace=True)
            
            # b) Select max_features WITHOUT replacement
            feature_indices = np.random.choice(n_features,
                                              size=self.max_features,
                                              replace=False)
            
            # Create bootstrap dataset using selected features
            X_bootstrap = dataset.X[bootstrap_indices][:, feature_indices]
            y_bootstrap = dataset.y[bootstrap_indices]
            
            bootstrap_dataset = Dataset(
                X=X_bootstrap,
                y=y_bootstrap,
                features=[dataset.features[i] for i in feature_indices] if dataset.features is not None else None,
                label=dataset.label
            )
            
            # 4. Create and train decision tree
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)
            
            # 5. Store (features, trained tree)
            self.trees.append((feature_indices, tree))
        
        # 7. Return self
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts labels using the tree ensemble (majority voting).
        
        Steps:
        1. For each tree:
           - Use only the selected features
           - Get predictions
        2. For each sample, apply majority vote
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to predict
        
        Returns
        -------
        np.ndarray
            Final predictions
        """
        if not self.trees:
            raise ValueError("Model must be trained before making predictions")
        
        n_samples = dataset.X.shape[0]
        
        # Store predictions of all trees
        # Shape: (n_estimators, n_samples)
        all_predictions = np.zeros((self.n_estimators, n_samples), dtype=object)
        
        # 1. Get predictions from each tree
        for idx, (feature_indices, tree) in enumerate(self.trees):
            # Select only features used by this tree
            X_subset = dataset.X[:, feature_indices]
            
            temp_dataset = Dataset(
                X=X_subset,
                y=dataset.y,
                features=[dataset.features[i] for i in feature_indices] if dataset.features is not None else None,
                label=dataset.label
            )
            
            predictions = tree.predict(temp_dataset)
            all_predictions[idx] = predictions
        
        # 2. Majority vote for each sample
        final_predictions = np.zeros(n_samples, dtype=object)
        
        for i in range(n_samples):
            votes = all_predictions[:, i]
            
            # Most frequent class
            vote_counts = Counter(votes)
            most_common_class = vote_counts.most_common(1)[0][0]
            
            final_predictions[i] = most_common_class
        
        return final_predictions
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes accuracy between true and predicted values.
        
        Parameters
        ----------
        dataset : Dataset
            Test dataset
        predictions : np.ndarray
            Predicted values
        
        Returns
        -------
        float
            Accuracy score
        """
        return accuracy(dataset.y, predictions)