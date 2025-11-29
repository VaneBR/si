import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from typing import Callable


class SelectPercentile(Transformer):
    """
    Selects features based on a percentile of the highest F-values.
    
    Parameters
    ----------
    score_func : Callable
        Scoring function to compute F and p values (default: f_classification)
    percentile : float
        Percentile for feature selection (0–100)
    
    Attributes
    ----------
    F : np.ndarray
        F-values for each feature
    p : np.ndarray
        p-values for each feature
    """
    
    def __init__(self, score_func: Callable = f_classification, percentile: float = 10):
        """
        Initializes SelectPercentile.
        
        Parameters
        ----------
        score_func : Callable
            Function to compute F and p values
        percentile : float
            Percentile for selection (0–100)
        """
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        
        super().__init__()
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    
    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Computes F-values and p-values for each feature.
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset
        
        Returns
        -------
        self : SelectPercentile
            Returns self
        """
        # Compute F and p values using the scoring function
        self.F, self.p = self.score_func(dataset)
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects a percentile of features based on F-values.
        
        This method ensures that the number of selected features corresponds
        to the specified percentile, handling ties at the threshold.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to transform
        
        Returns
        -------
        Dataset
            Dataset with selected features
        """
        if self.F is None:
            raise ValueError("The model must be fitted before calling transform")
        
        # Compute number of features to select
        n_features = dataset.X.shape[1]
        n_select = int(np.ceil(n_features * self.percentile / 100))
        
        # Ensure at least 1 feature and at most all
        n_select = max(1, min(n_select, n_features))
        
        # Compute threshold based on percentile
        # percentile=40 means selecting top 40%, so threshold at the 60th percentile
        threshold_percentile = 100 - self.percentile
        threshold = np.percentile(self.F, threshold_percentile)
        
        # Initial mask: features with F > threshold
        mask = self.F > threshold
        n_selected = np.sum(mask)
        
        # If not enough features are selected, include features equal to threshold
        if n_selected < n_select:
            # Find features exactly at the threshold (ties)
            tied_features = np.where(self.F == threshold)[0]
            
            # Number of additional features needed
            n_needed = n_select - n_selected
            
            # Add the first n_needed tied features
            if len(tied_features) > 0:
                for i in range(min(n_needed, len(tied_features))):
                    mask[tied_features[i]] = True
        
        # If too many features are selected, use top-k directly
        elif n_selected > n_select:
            # Get the indices of the top k F-values
            top_k_indices = np.argsort(self.F)[-n_select:]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k_indices] = True
        
        # Apply mask
        X_selected = dataset.X[:, mask]
        
        # Update feature names if they exist
        features_selected = None
        if dataset.features is not None:
            features_selected = [dataset.features[i] for i in range(len(mask)) if mask[i]]
        
        # Return new dataset
        return Dataset(
            X=X_selected, 
            y=dataset.y, 
            features=features_selected, 
            label=dataset.label
        )
    
    def _get_feature_mask(self) -> np.ndarray:
        """
        Returns the boolean mask of the selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating selected features
        """
        if self.F is None:
            raise ValueError("The model must be fitted first")
        
        n_features = len(self.F)
        n_select = int(np.ceil(n_features * self.percentile / 100))
        n_select = max(1, min(n_select, n_features))
        
        threshold_percentile = 100 - self.percentile
        threshold = np.percentile(self.F, threshold_percentile)
        
        mask = self.F > threshold
        n_selected = np.sum(mask)
        
        if n_selected < n_select:
            tied_features = np.where(self.F == threshold)[0]
            n_needed = n_select - n_selected
            
            if len(tied_features) > 0:
                for i in range(min(n_needed, len(tied_features))):
                    mask[tied_features[i]] = True
        
        elif n_selected > n_select:
            top_k_indices = np.argsort(self.F)[-n_select:]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k_indices] = True
        
        return mask