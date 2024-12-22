import numpy as np
import pandas as pd

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
    numerator = ((yhat - mu) ** 2).sum()
    
    # Calculate the denominator TSS
    denominator = ((y - mu) ** 2).sum()
    
    # Compute the out-of-sample R2
    r2 = numerator / denominator
    
    return r2

def oos_residuals(y, x, beta, first, last):
    """
    Compute out-of-sample residuals for a given range of dates.

    Parameters:
    y (pd.Series): A pandas Series with shape T (outcome variable) and a DatetimeIndex.
    x (pd.DataFrame): A pandas DataFrame with shape T by k (regressors) and a DatetimeIndex matching `y`.
    beta (pd.Series): A pandas Series with shape k (regression coefficients) and index matching columns of `x`.
    first (str): Start date (inclusive) in string format (e.g., "1970" or "1974-03-01").
    last (str): End date (inclusive) in string format (e.g., "1980" or "1979-03-01").

    Returns:
    pd.Series: Residuals (y - Xβ) for the sample bracketed by `first` and `last`.
    """
    # Slice the data to include only the rows between `first` and `last`
    y_sample = y[first:last]
    x_sample = x[first:last]
    
    # Calculate predicted values (Xβ)
    y_pred = x_sample.dot(beta)
    
    # Compute residuals (y - Xβ)
    resid = y_sample - y_pred
    
    return resid