import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

# models/ridge_regression_least_squares.py

class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the Least Squares closed-form solution.
    
    This implementation uses the analytical solution for Ridge Regression
    instead of Gradient Descent, computing the optimal coefficients directly using:
    
    θ = (X^T X + λI)^(-1) X^T y
    
    where λ is the L2 regularization parameter.
    
    Parameters
    ----------
    l2_penalty : float
        L2 regularization parameter (λ) (default: 1.0)
    scale : bool
        Whether to scale the data (Z-score normalization) (default: True)
    
    Attributes
    ----------
    theta : np.ndarray
        Model coefficients for each feature (excluding intercept)
    theta_zero : float
        Intercept coefficient (not regularized)
    mean : np.ndarray
        Mean of each feature (used for scaling)
    std : np.ndarray
        Standard deviation of each feature (used for scaling)
    """
    
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        """
        Initializes the RidgeRegressionLeastSquares model.
        
        Parameters
        ----------
        l2_penalty : float
            L2 regularization parameter (default: 1.0)
        scale : bool
            Whether to scale the data (default: True)
        """
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale
        
        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
    
    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates theta and theta_zero using the Least Squares solution.
        
        Steps:
        1. Scale data if necessary
        2. Add intercept term (column of 1s)
        3. Create penalty matrix (λI)
        4. Zero out first element (do not penalize intercept)
        5. Compute parameters: θ = (X^T X + λI)^(-1) X^T y
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset
        
        Returns
        -------
        self : RidgeRegressionLeastSquares
            Returns self
        """
        if dataset.y is None:
            raise ValueError("Dataset must have labels (y) for training")
        
        X = dataset.X.copy()
        y = dataset.y.copy()
        
        # 1. Scale data if necessary
        if self.scale:
            # Compute and store mean and std
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            
            # Avoid division by zero
            self.std[self.std == 0] = 1.0
            
            # Z-score normalization
            X = (X - self.mean) / self.std
        
        # 2. Add intercept term (column of 1s in the first position)
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        # 3. Create penalty matrix (λI)
        n_features = X_with_intercept.shape[1]
        penalty_matrix = self.l2_penalty * np.eye(n_features)
        
        # 4. Do not penalize the intercept
        penalty_matrix[0, 0] = 0
        
        # 5. Compute parameters using the Ridge Regression formula
        
        # Compute X^T X
        XtX = X_with_intercept.T.dot(X_with_intercept)
        
        # Add penalty: X^T X + λI
        XtX_plus_penalty = XtX + penalty_matrix
        
        # Compute inverse: (X^T X + λI)^(-1)
        XtX_inv = np.linalg.inv(XtX_plus_penalty)
        
        # Compute X^T y
        Xty = X_with_intercept.T.dot(y)
        
        # Compute θ = (X^T X + λI)^(-1) X^T y
        theta_all = XtX_inv.dot(Xty)
        
        # Separate intercept and coefficients
        self.theta_zero = theta_all[0]
        self.theta = theta_all[1:]
        
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts y using the estimated coefficients.
        
        Steps:
        1. Scale data if necessary (using training mean/std)
        2. Add intercept term
        3. Compute y_pred = X @ θ
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to predict
        
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.theta is None or self.theta_zero is None:
            raise ValueError("Model must be trained before predicting")
        
        X = dataset.X.copy()
        
        # 1. Scale if needed (use training mean/std)
        if self.scale:
            if self.mean is None or self.std is None:
                raise ValueError("Model was not trained with scaling")
            X = (X - self.mean) / self.std
        
        # 2. Add intercept
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        # 3. Concatenate intercept + coefficients
        theta_all = np.r_[self.theta_zero, self.theta]
        
        # 4. Compute predictions
        y_pred = X_with_intercept.dot(theta_all)
        
        return y_pred
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
    """
    Computes MSE between true and predicted values.
    
    Parameters
    ----------
    dataset : Dataset
        Test dataset
    predictions : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MSE (Mean Squared Error)
    """
    # Compute MSE using provided predictions
    return mse(dataset.y, predictions)