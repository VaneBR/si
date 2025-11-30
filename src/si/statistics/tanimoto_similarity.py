import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Tanimoto similarity/distance between one binary sample
    and multiple binary samples.
    
    The Tanimoto similarity (also known as Jaccard) for binary data is:
    T(A,B) = (A ∩ B) / (A ∪ B)
    
    The Tanimoto distance is: 1 - T(A,B)
    
    Parameters
    ----------
    x : np.ndarray
        A single binary sample (1D array)
    y : np.ndarray
        Multiple binary samples (2D array), where each row is a sample
    
    Returns
    -------
    np.ndarray
        Array containing the Tanimoto distances between x and each sample in y
    """
    # Ensure that x is 1D and y is 2D
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if y.ndim != 2:
        raise ValueError("y must be a 2D array")
    if x.shape[0] != y.shape[1]:
        raise ValueError("x and y must have the same number of features")
    
    # Convert to boolean arrays to ensure binary values
    x = x.astype(bool)
    y = y.astype(bool)
    
    # Compute intersection (A ∩ B) - both are 1
    # Equivalent to logical AND
    intersection = np.sum(x & y, axis=1)
    
    # Compute union (A ∪ B) - at least one is 1
    # Equivalent to logical OR
    union = np.sum(x | y, axis=1)
    
    # Compute Tanimoto similarity
    # Avoid division by zero
    similarity = np.where(union > 0, intersection / union, 0.0)
    
    # Return the distance (1 - similarity)
    distance = 1 - similarity
    
    return distance