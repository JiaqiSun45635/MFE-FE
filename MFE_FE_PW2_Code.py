import numpy as np
import pandas as pd

def oos_rsquared(y,yhat,mu):
# Out-of-sample R2
# Produce a function that will compute the out-of-sample R2.
# r2 = oos_rsquared(y,yhat,mu)
#Outputs:
## r2: The out-of-sample R2 (float)
# Inputs:
## y: A pandas Series with shape n. The out-of-sample realised value.
## yhat: A pandas Series with shape n. The forecasts of y. The index of yhat will match that of y,
### so that observation i of yhat will be the forecast of y in position i.
## mu: A float. The in-sample mean of the time-series. Essentially a forecast of y assuming that the
### correct model has a constant mean.
# return "something"

def oos_residuals(y, x, beta, first, last):
# Out-of-Sample Residual Construction
# Compute out-of-sample residuals for values stored in a Series where regressors are a DataFrame and
### parameters are a Series.
# resid = oos_residuals(y, x, beta, first, last)
# Outputs:
## resid: A pandas Series with shape n. The value of Y − Xbβ for the relevant sample.
# Inputs:
## y: A pandas Series with shape T . y will have a DatetimeIndex.
## x: A pandas DataFrame with shape T by k. The index of x will match y.
## beta: A pandas Series with shape k. The regression coefficients. The index of beta will match the
### column names of x.
## first: A date in string format, e.g. "1970" or "1974-03-01".
##last: A date in string format, e.g. "1980" or "1979-03-01".
# Note: You should return the residuals only for the sample bracketed by first and last (inclusive).
# return w[0], w[1], z