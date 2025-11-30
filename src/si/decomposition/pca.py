import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition 
    of the covariance matrix.
    
    PCA is a linear algebra technique used to reduce the dimensionality
    of a dataset while retaining as much variance as possible.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to keep
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PCA.
        
        Parameters
        ----------
        n_components : int
            Number of principal components (default: 2)
        """
        super().__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Estimate the mean, principal components, and explained variance.
        
        Steps:
        1. Center the data (compute mean and subtract it)
        2. Compute covariance matrix
        3. Eigenvalue decomposition of the covariance matrix
        4. Select the largest n_components eigenvalues and eigenvectors
        5. Compute explained variance
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset
        
        Returns
        -------
        self : PCA
            Returns self
        """
        # 1. Compute and store the mean
        self.mean = np.mean(dataset.X, axis=0)
        
        # Center data
        X_centered = dataset.X - self.mean
        
        # 2. Compute covariance matrix
        # rowvar=False because each column is a variable
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # 3. Eigenvalue decomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Ensure eigenvalues are real numbers
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select the n largest components
        self.components = eigenvectors[:, :self.n_components].T
        
        # 5. Compute explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset into the principal component space.
        
        Steps:
        1. Center the data using the training mean
        2. Project onto the principal components: X_reduced = X_centered @ components.T
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to transform
        
        Returns
        -------
        Dataset
            Transformed dataset with reduced dimensionality
        """
        if self.mean is None or self.components is None:
            raise ValueError("The model must be fitted before transforming")
        
        # 1. Center data using training mean
        X_centered = dataset.X - self.mean
        
        # 2. Project onto principal components
        X_reduced = np.dot(X_centered, self.components.T)
        
        # Create new feature names
        features = [f'PC{i+1}' for i in range(self.n_components)]
        
        return Dataset(
            X=X_reduced, 
            y=dataset.y, 
            features=features, 
            label=dataset.label
        )