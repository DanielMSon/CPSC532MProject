import numpy as np
import pandas as pd


# Standardize each column with mean 0 and standard deviation 1
def standardize_cols(X, mu=None, sigma=None):
    X_values = X.values  # convert pandas dataframe into numpy array

    if mu is None:
        mu = np.mean(X_values, axis=0)

    if sigma is None:
        sigma = np.std(X_values, axis=0)
        sigma[sigma < 1e-8] = 1.

    X = pd.DataFrame((X_values - mu) / sigma, columns=X.columns)  # convert numpy array into pandas dataframe

    return X, mu, sigma

# Standardize dataset. For validation dataset, take mean and standard deviation of each column from training dataset to
# preserve golden rule. Adds a bias column (column of all 1) if include_y_int=True. Separate dataset into set of
# features and label vector
def standardize_dataset(data, data_valid, include_y_int=True):
    data, mu, sigma = standardize_cols(data)
    data_valid, _, _ = standardize_cols(data_valid, mu=mu, sigma=sigma)

    # add a y-intercept basis to data
    if include_y_int is True:
        DFYint = pd.DataFrame(np.ones((data.shape[0], 1)), columns=['y-intercept'])
        data = pd.concat([DFYint, data], axis=1)

        DFYint_valid = pd.DataFrame(np.ones((data_valid.shape[0], 1)), columns=['y-intercept'])
        data_valid = pd.concat([DFYint_valid, data_valid], axis=1)

    y = data['SalePrice']
    X = data.drop('SalePrice', axis=1)

    y_valid = data_valid['SalePrice']
    X_valid = data_valid.drop('SalePrice', axis=1)

    return X, y, X_valid, y_valid
