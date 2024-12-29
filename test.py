import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def generate_dataset_with_target_r2(original_data, target_r2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Ensure original_data is a pandas Series
    if not isinstance(original_data, pd.Series):
        original_data = pd.Series(original_data)
    
    # Calculate the mean and standard deviation of the original data
    original_mean = original_data.mean()
    original_std = original_data.std()
    
    # Normalize the original data
    normalized_original = (original_data - original_mean) / original_std
    
    # Generate noise with mean 0 and standard deviation 1
    noise = np.random.normal(loc=0, scale=1, size=original_data.shape)
    noise = pd.Series(noise, index=original_data.index)
    
    # Create the new dataset
    new_data = (np.sqrt(target_r2) * normalized_original +
                np.sqrt(1 - target_r2) * noise)
    
    # Rescale the new data to match the original scale
    new_data = new_data * original_std + original_mean
    
    # Verify R^2
    calculated_r2 = r2_score(original_data, new_data)
    print(f"Target R^2: {target_r2}, Calculated R^2: {calculated_r2}")
    print(f"Original Mean: {original_mean}")
    
    return new_data, original_mean

def oos_rsquared(y,yhat,mu):
    """
    Compute the out-of-sample R2.

    Parameters:
    y (pd.Series): Out-of-sample realized values, shape (n,).
    yhat (pd.Series): Forecasts of y, shape (n,). Indices should match y.
    mu (float): In-sample mean of the time-series.

    Returns:
    float: The out-of-sample R2.
    """
    # Calculate the numerator RSS
    numerator = ((y - yhat) ** 2).sum()
    
    # Calculate the denominator TSS
    denominator = ((y - mu) ** 2).sum()
    
    # Compute the out-of-sample R2
    r2 = 1 - numerator / denominator
    
    return r2

# Example usage
y = pd.Series(np.random.rand(100))
target_r2 = 0.8  # Desired R^2 value
yhat, mu = generate_dataset_with_target_r2(y, target_r2, random_seed=42)

print(oos_rsquared(y,yhat,mu))