import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    RMSE is a common regression metric that measures the square root of the 
    mean of squared errors. Lower values indicate better performance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True y values
    y_pred : np.ndarray
        Predicted y values
    
    Returns
    -------
    float
        RMSE between y_true and y_pred
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty")
    
    # Compute differences
    differences = y_true - y_pred
    
    # Square them
    squared_differences = differences ** 2
    
    # Compute mean
    mean_squared_error = np.mean(squared_differences)
    
    # Compute square root
    root_mean_squared_error = np.sqrt(mean_squared_error)
    
    return root_mean_squared_error