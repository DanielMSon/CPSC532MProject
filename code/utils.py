import numpy as np
import pandas as pd

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    # n_rows, n_cols = X.shape

    X_values = X.values

    if mu is None:
        mu = np.mean(X_values, axis=0)

    if sigma is None:
        sigma = np.std(X_values, axis=0)
        sigma[sigma < 1e-8] = 1.

    X = pd.DataFrame((X_values - mu) / sigma, columns=X.columns)

    return X, mu, sigma


def standardize_dataset(data, include_y_int=True):
    # Load and standardize the data and add the bias term

    data, mu, sigma = standardize_cols(data)

    # add a y-intercept basis to data
    if include_y_int is True:
        DFYint = pd.DataFrame(np.ones((data.shape[0], 1)), columns=['y-intercept'])
        data = pd.concat([DFYint, data], axis=1)

    y = data['SalePrice']
    X = data.drop('SalePrice', axis=1)

    return X, y
