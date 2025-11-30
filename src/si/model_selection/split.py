from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, 
                                test_size: float = 0.2, 
                                random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and test sets while keeping the class 
    proportions (stratified split).
    
    This function ensures that the class distribution in the training and test 
    sets is approximately the same as in the original dataset.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset to be split
    test_size : float
        Proportion of the dataset to be used as test (e.g., 0.2 for 20%)
    random_state : int, optional
        Seed for generating random permutations
    
    Returns
    -------
    Tuple[Dataset, Dataset]
        Tuple containing (train_dataset, test_dataset)
    """
    if dataset.y is None:
        raise ValueError("Dataset must have labels (y) for stratified split")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Set seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)
    
    # Initialize empty lists for train and test indices
    train_indices = []
    test_indices = []
    
    # Loop through unique classes
    for class_label in unique_classes:
        # Get indices of all samples of this class
        class_indices = np.where(dataset.y == class_label)[0]
        
        # Compute number of test samples for this class
        n_test_samples = int(len(class_indices) * test_size)
        
        # Ensure at least 1 test sample if the class exists
        if n_test_samples == 0 and len(class_indices) > 0:
            n_test_samples = 1
        
        # Shuffle indices for this class
        shuffled_indices = np.random.permutation(class_indices)
        
        # Select indices for test and train
        test_indices_class = shuffled_indices[:n_test_samples]
        train_indices_class = shuffled_indices[n_test_samples:]
        
        # Add to global index lists
        test_indices.extend(test_indices_class)
        train_indices.extend(train_indices_class)
    
    # Convert lists to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle the final indices to avoid class-ordered grouping
    train_indices = np.random.permutation(train_indices)
    test_indices = np.random.permutation(test_indices)
    
    # Create training and test datasets
    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]
    
    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]
    
    train_dataset = Dataset(X=X_train, y=y_train, 
                           features=dataset.features, 
                           label=dataset.label)
    
    test_dataset = Dataset(X=X_test, y=y_test, 
                          features=dataset.features, 
                          label=dataset.label)
    
    # Return datasets
    return train_dataset, test_dataset

