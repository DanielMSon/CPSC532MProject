import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import KFold, cross_val_score, train_test_split

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

def evaluate_model(model, X, y, cross_val=False, valid_size=0.1, n_splits=10, shuffle_data=True, random_state=2, verbose=False):
    """ Evaluate model by splitting dataset into training and validation 
    sets and report both training and validation errors
    
    Arguments:
        model {model} -- regression/classificaion model
        X {ndarray} -- n x d matrix
        y {ndarray} -- n x 1 array of correct labels
    
    Keyword Arguments:
        cross_val {bool} -- whether cross validation will be used (default: {False})
        valid_size {float} -- only used when cross_val is False. Split dataset into (1-valid_size)*n for training and valid_size*n for validation
        n_splits {int} -- only used when cross_val is True, will split dataset into n_splits folds for cross validation      
                        Must divide 1.0 into an integer. (default: {10})
        shuffle_data {bool} -- whether to shuffle data at the beginning (default: {True})
        verbose {bool} -- whether to print out training and validation error (default: {False})
    
    Returns:
        tuple -- err_tr, err_va
    """

    # initialize variables
    if (cross_val == True):
        errs_tr = np.array([])
        errs_va = np.array([])

    err_tr = 0
    err_va = 0

    # print input X size
    # if (verbose == True):
        # n, d = np.shape(X)
        # print("X shape: {} x {}".format(str(n), str(d)))

    # shuffle data
    if (shuffle_data == True):
        X, y = shuffle(X, y, random_state=random_state)

    if (cross_val == True):
        # split data for KFold and change n_folds 
        kf = KFold(n_splits=n_splits)

        # evaluate with cross validation
        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            temp_err_tr, temp_err_va = _evaluate(model, X_train, y_train, X_valid, y_valid)
            
            # save errors
            errs_tr = np.append(errs_tr, temp_err_tr)
            errs_va = np.append(errs_va, temp_err_va)
            
        err_tr = np.mean(errs_tr)
        err_va = np.mean(errs_va)

    else:
        # split data into training and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=random_state)
        err_tr, err_va = _evaluate(model, X_train, y_train, X_valid, y_valid)
    
    if (verbose == True):
        print("training error: {0:.6g}".format(err_tr))
        print("validation error: {0:.6g}".format(err_va))

    return err_tr, err_va

def _evaluate(model, X_train, y_train, X_valid=None, y_valid=None):
    """ helper function for evaluate model
    
    Arguments:
        model {model object} -- regression model
        X_train {ndarray} -- training set
        y_train {ndarray} -- training labels
    
    Keyword Arguments:
        X_valid {ndarray} -- validation set (default: {None})
        y_valid {ndarray} -- validation labels (default: {None})
    
    Returns:
        err_tr, err_va {tuple} -- err_va is None if either X_valid or y_valid is None
    """

    err_tr = 0
    err_va = 0
    
    model.fit(X_train, y_train)
    y_hat = model.predict(X_train)
    err_tr = np.mean((y_hat-y_train)**2)

    if X_valid is not None and y_valid is not None:
        y_hat = model.predict(X_valid)
        err_va = np.mean((y_hat - y_valid)**2)
    else:
        err_va = None 

    return err_tr, err_va