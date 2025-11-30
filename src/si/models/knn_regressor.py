import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse
from typing import Callable


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor.
    
    The KNN algorithm for regression estimates the value for a sample
    based on the average of the k most similar training examples.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider (default: 3)
    distance : Callable
        Function that computes the distance between a sample and the
        training dataset (default: euclidean_distance)
    
    Attributes
    ----------
    dataset : Dataset
        Stored training dataset
    """
    
    def __init__(self, k: int = 3, distance: Callable = euclidean_distance):
        """
        Initializes the KNNRegressor.
        
        Parameters
        ----------
        k : int
            Number of neighbors (default: 3)
        distance : Callable
            Distance function (default: euclidean_distance)
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None
    
    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.
        
        In KNN, there is no traditional training — we simply store
        the data for later use during prediction (lazy learning).
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset
        
        Returns
        -------
        self : KNNRegressor
            Returns self
        """
        if dataset.y is None:
            raise ValueError("Dataset must have labels (y) for regression")
        
        # Store training dataset
        self.dataset = dataset
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates values for samples based on the k nearest neighbors.
        
        For each sample:
        1. Compute distances to all training samples
        2. Find the k nearest neighbors
        3. Compute the mean of their y values
        
        Parameters
        ----------
        dataset : Dataset
            Test dataset
        
        Returns
        -------
        np.ndarray
            Array with predicted values (y_pred)
        """
        if self.dataset is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Array to store predictions
        predictions = np.zeros(dataset.X.shape[0])
        
        # For each sample in test dataset
        for i, sample in enumerate(dataset.X):
            # 1. Compute distances to all training samples
            distances = self.distance(sample, self.dataset.X)
            
            # 2. Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            # 3. Get y values of closest neighbors
            k_nearest_values = self.dataset.y[k_nearest_indices]
            
            # 4. Compute mean value
            predicted_value = np.mean(k_nearest_values)
            
            # Store prediction
            predictions[i] = predicted_value
        
        return predictions
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the RMSE score for the predictions.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset with true labels
        predictions : np.ndarray
            Predicted values
        
        Returns
        -------
        float
            RMSE score (lower is better)
        """
        return rmse(dataset.y, predictions)