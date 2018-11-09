import argparse
import numpy as np
import pandas as pd
from utils import standardize_dataset
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question

    # Show top 20 features that has the highest proportion of null (or 'NA') entries.
    if question == "missing_val":
        X_train = pd.read_csv('../data/train.csv')

        # show features with high proportion of null entries
        train_data_na = (X_train.isnull().sum() / len(X_train)) * 100
        train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': train_data_na})
        print(missing_data.head(20))  # print top 20 features with significant null entries

    # Fill in missing entries and change categorical into numerical features. Standardization not included
    elif question == "pre_processing":
        X_train = pd.read_csv('../data/train.csv')

        print("Size before pre-processing: ", X_train.shape)

        # Fill in missing (or null entries)
        # region NULL_ENTRY_CLEAN_UP
        # entries of units without these feature is NA. State "None" to say these features don't exist
        for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):
            X_train[col] = X_train[col].fillna('None')

        # units in same neighbourhood should have similar lot frontage. Assign median of its neighbourhood to unit
        X_train["LotFrontage"] = X_train.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))

        # units with no garage has these features as NA. Set to "None"
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            X_train[col] = X_train[col].fillna('None')

        # units with no garage has this feature as NA. Set the year to 0. Although may need to classify this
        # categorically somehow instead
        X_train['GarageYrBlt'] = X_train['GarageYrBlt'].fillna(0)

        # units with no basement has this feature as NA. Set to "None"
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            X_train[col] = X_train[col].fillna('None')

        # assume missing entries due to no masonry veneer existing. Set type as "None" and area as 0
        X_train["MasVnrType"] = X_train["MasVnrType"].fillna("None")
        X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)

        # drop remaining rows with "NA" entry
        print("Size before dropping rows with nan entries: ", X_train.shape)
        X_train = X_train.dropna()
        print("Size after dropping rows with nan entries: ", X_train.shape)
        # endregion

        # Turn categorical features' unique entries into a binary features of its own
        # region HOT_ENCODE
        cols = ('MSSubClass', 'MSZoning', 'Alley', 'LotConfig', 'LandContour', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                'Heating', 'Electrical', 'GarageType', 'Utilities', 'Neighborhood', 'MiscFeature', 'SaleType',
                'SaleCondition', 'Fence')
        for c in cols:
            X_train = pd.concat([X_train, pd.get_dummies(X_train[c], prefix=c)], axis=1)
            X_train.drop(c, axis=1, inplace=True)

        print("Size after hot encoding: ", X_train.shape)
        # endregion

        # Turn categorical features' unique entries into int. Do this for features where ranking in order has meaning
        # region LABEL_ENCODE
        mapper = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        cols = ('ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                'GarageQual', 'GarageCond', 'PoolQC')
        for c in cols:
            X_train[c] = X_train[c].replace(mapper)

        mapper = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        cols = ('BsmtFinType1', 'BsmtFinType2')
        for c in cols:
            X_train[c] = X_train[c].replace(mapper)

        mapper = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
        X_train['Functional'] = X_train['Functional'].replace(mapper)

        mapper = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        X_train['BsmtExposure'] = X_train['BsmtExposure'].replace(mapper)

        mapper = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        X_train['GarageFinish'] = X_train['GarageFinish'].replace(mapper)

        mapper = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
        X_train['LandSlope'] = X_train['LandSlope'].replace(mapper)

        mapper = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
        X_train['LotShape'] = X_train['LotShape'].replace(mapper)

        mapper = {'N': 0, 'P': 1, 'Y': 2}
        X_train['PavedDrive'] = X_train['PavedDrive'].replace(mapper)

        mapper = {'Grvl': 0, 'Pave': 1}
        X_train['Street'] = X_train['Street'].replace(mapper)

        mapper = {'N': 0, 'Y': 1}
        X_train['CentralAir'] = X_train['CentralAir'].replace(mapper)
        # endregion

        X_train.to_csv('../data/train_preprocessed.csv', index=False)

    # Over varying lambda, bolasso to select features and calculate cross validated least square (with only selected
    # features) as score. Use to find optimal lambda
    elif question == "feature_select_lambda":
        data = pd.read_csv('../data/train_preprocessed.csv')

        # hyper-parameters
        lammy_max = 0.1  # max lambda value to try up to (not inclusive). Start from 0
        lammy_interval = 0.005  # lambda value interval to increase each iteration
        n_bootstraps = 20  # number of bootstrap samples to generate for feature selection
        n_splits = 20  # number of cross validation folds to perform when calculating score for each lambda

        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(data)
        for lammy in np.arange(0, lammy_max, lammy_interval):
            # bootstrap and calculate weights
            X_train, y_train, _, _ = standardize_dataset(data, data)
            coef_matrix = np.zeros((n_bootstraps, X_train.shape[1]))  # make matrix to store each bootstrap lasso weight
            for i in range(0, n_bootstraps):
                X_boot, y_boot = resample(X_train, y_train, n_samples=X_train.shape[0])  # make bootstrap data set
                model = Lasso(alpha=lammy, fit_intercept=False)
                model.fit(X_boot, y_boot)
                coef_matrix[i, :] = model.coef_

            # detect feature weights that don't meet 90% intersection
            col_drop_list = []
            for c in range(0, X_train.shape[1]):
                non_zero = np.count_nonzero(coef_matrix[:, c])
                if non_zero < coef_matrix.shape[0]*0.9:  # check if weight is significant less than 90% of bootstrap runs
                    col_drop_list.append(c)
            X_train.drop(X_train.columns[col_drop_list], axis=1, inplace=True)  # drop insignificant features

            # run cross validation to evaluate E_valid only considering significant features. Use un-regularized least
            # square model to fit for weights and evaluate E_valid
            data_input = pd.concat([X_train, y_train], axis=1)
            E_train_list = []
            E_valid_list = []
            for train_index, valid_index in kf.split(data):
                X_train, X_valid = data_input.iloc[train_index], data_input.iloc[valid_index]
                X_train, y_train, X_valid, y_valid = standardize_dataset(X_train, X_valid)

                # model = Lasso(alpha=0, fit_intercept=False)
                model = LinearRegression(fit_intercept=False)
                model.fit(X_train, y_train)
                E_train_list.append(np.mean(abs(model.predict(X_train) - y_train)))
                E_valid_list.append(np.mean(abs(model.predict(X_valid) - y_valid)))

            # calculate average of errors over cross validation
            E_train = sum(E_train_list)/len(E_train_list)
            E_valid = sum(E_valid_list)/len(E_valid_list)
            print("For lambda = %0.3f, E_train: %0.3f, E_valid: %0.3f, E_approx: %0.3f" % (lammy, E_train, E_valid,
                                                                                           E_valid - E_train))

    # Run more extensive bolasso over given lambda value. Prints features to keep and drop. Saves data set with only
    # significant features (and y or SalePrice) as a csv.
    elif question == "feature_select":
        data = pd.read_csv('../data/train_preprocessed.csv')

        # hyper-parameters
        lammy = 0.01
        n_bootstraps = 200

        X_train, y_train, _, _ = standardize_dataset(data, data)
        coef_matrix = np.zeros((n_bootstraps, X_train.shape[1]))

        for i in range(0, n_bootstraps):
            X_boot, y_boot = resample(X_train, y_train, n_samples=X_train.shape[0])
            model = Lasso(alpha=lammy, fit_intercept=False)
            model.fit(X_boot, y_boot)
            coef_matrix[i, :] = model.coef_

        # detect feature weights that don't meet 90% intersection
        col_drop_list = []
        label_name_list = list(X_train)
        for c in range(0, X_train.shape[1]):
            non_zero = np.count_nonzero(coef_matrix[:, c])
            if non_zero < coef_matrix.shape[0] * 0.9:
                print("column %i or %s should be dropped" % (c, label_name_list[c]))
                col_drop_list.append(c)
        X_train.drop(X_train.columns[col_drop_list], axis=1, inplace=True)  # drop the features that aren't significant
        label_name_list = list(X_train)

        # save un-standardized data set with only significant features (this has X and y).
        DFYint = pd.DataFrame(np.ones((data.shape[0], 1)), columns=['y-intercept'])
        data = pd.concat([DFYint, data], axis=1)
        SalePriceCol = data['SalePrice']
        data.drop(labels=['SalePrice'], axis=1, inplace=True)
        data.drop(data.columns[col_drop_list], axis=1, inplace=True)
        data.insert(0, 'SalePrice', SalePriceCol)
        data.to_csv('../data/train_sig_features.csv', index=False)

        # report
        print("Number of features dropped: ", len(col_drop_list))
        print("Number of features kept: ", len(label_name_list))
        print("These are the features kept: ")
        for feature in label_name_list:
            print(feature)







