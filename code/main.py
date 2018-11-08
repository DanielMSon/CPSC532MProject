import argparse
import numpy as np
import pandas as pd
from utils import standardize_dataset
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "missing_val":
        X_train = pd.read_csv('../data/train.csv')

        # show features with high proportion of null entries
        train_data_na = (X_train.isnull().sum() / len(X_train)) * 100
        train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': train_data_na})
        print(missing_data.head(20))  # print top 20 features with significant null entries

    elif question == "fill_missing_val":
        X_train = pd.read_csv('../data/train.csv')

        print("Size before pre-processing: ", X_train.shape)

        # region NULL_ENTRY_CLEAN_UP
        # state "None" to say these features don't exist
        for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):
            X_train[col] = X_train[col].fillna('None')

        # houses in same neighbourhood should have similar lot frontage. Assign median of neighbourhood
        X_train["LotFrontage"] = X_train.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))

        # houses with no garage has these features as NA. Set to "None"
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            X_train[col] = X_train[col].fillna('None')

        # houses with no garage has this feature as NA. Set the year to 0. Although may need to classify this
        # categorically somehow instead
        X_train['GarageYrBlt'] = X_train['GarageYrBlt'].fillna(0)

        # houses with no basement has this feature as NA. Set to "None"
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            X_train[col] = X_train[col].fillna('None')

        # missing entries. Assume that no masonry veneer exists. Set type as "None" and area as 0
        X_train["MasVnrType"] = X_train["MasVnrType"].fillna("None")
        X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)

        # drop remaining rows with "NA" entry
        print("Size before dropping rows with nan entries: ", X_train.shape)
        X_train = X_train.dropna()
        print("Size after dropping rows with nan entries: ", X_train.shape)
        # endregion

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

    elif question == "feature_select_lambda":
        data = pd.read_csv('../data/train_preprocessed.csv')

        lammy_max = 0.5
        lammy_interval = 0.01

        n_splits = 10
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(data)
        for lammy in np.arange(0, lammy_max, lammy_interval):
            E_train_list = []
            E_valid_list = []
            for train_index, valid_index in kf.split(data):
                X_train, X_valid = data.iloc[train_index], data.iloc[valid_index]
                X_train, y_train, X_valid, y_valid = standardize_dataset(X_train, X_valid)

                model = Lasso(alpha=lammy)
                model.fit(X_train, y_train)
                E_train_list.append(np.mean(abs(model.predict(X_train) - y_train)))
                E_valid_list.append(np.mean(abs(model.predict(X_valid) - y_valid)))

                #print("for lambda=", lammy, " ", list(zip(model.coef_, X_train.columns)))

            E_train = sum(E_train_list)/len(E_train_list)
            E_valid = sum(E_valid_list)/len(E_valid_list)
            print("For lambda = %0.2f, E_train: %0.3f, E_valid: %0.3f, E_approx: %0.3f" %(lammy, E_train, E_valid, E_valid-E_train))

        # print(list(zip(model.coef_, X_train.columns)))
        # weightArray = pd.DataFrame(model.coef_, columns=X_train.columns)
        # print(weightArray)

        # X_boot, y_boot = resample(X_train, y_train, n_samples=X_train.shape[0])  # make bootstrap sample
        # model = Lasso(alpha=0.1)
        # model.fit(X_boot, y_boot)
        # E_train = np.sum(abs(model.predict(X_boot) - y_boot))
        # E_valid = np.sum(abs(model.predict(X_train)-y_train))





