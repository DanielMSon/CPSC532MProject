import argparse
import numpy as np
import pandas as pd
import os

from utils import standardize_dataset, evaluate_model
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

##ANOVA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression, mutual_info_regression
from sklearn import preprocessing

#XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from models import NeuralNetRegressor, neuralNetHyperparamTuning, AveragingRegressor, StackingRegressor, StackingAveragedModels
import lightgbm as lgb

# global constant/var
dataset = 'train'

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
        X_train = pd.read_csv('../data/{}.csv'.format(dataset))

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

        # Correct the skew on target variable and features to make it more normally distributed for better linear model
        # prediction. Placed after label encode but before hot encode to regard ordinal feature but not categorical
        # feature.
        # region SKEW_CORRECTION
        # Correct target skew by using log(1+x)
        X_train["SalePrice"] = np.log1p(X_train["SalePrice"])  # FIXME:remember to cancel out the skew correct after prediction by using np.expm1(X_train["SalePrice"])

        # # Check the new distribution
        # sns.distplot(X_train['SalePrice'], fit=norm);
        # # Get the fitted parameters used by the function
        # (mu, sigma) = norm.fit(X_train['SalePrice'])
        # print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        # # Now plot the distribution
        # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
        #            loc='best')
        # plt.ylabel('Frequency')
        # plt.title('SalePrice distribution')
        # # Get also the QQ-plot
        # fig = plt.figure()
        # res = stats.probplot(X_train['SalePrice'], plot=plt)
        # plt.show()

        # correcting feature skew
        numeric_feats = X_train.dtypes[X_train.dtypes != "object"].index

        skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False) # Check the skew of all numerical features
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_feats})
        print(skewness.head(30))

        skewness = skewness[abs(skewness) > 0.75]
        skewness.dropna(inplace=True)
        print("There are {} skewed numerical features to correct".format(skewness.shape[0]))

        X_train[skewness.index] = np.log1p(X_train[skewness.index])
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

        X_train.to_csv('../data/{}_preprocessed.csv'.format(dataset), index=False)

    # Over varying lambda, bolasso to select features and calculate cross validated least square (with only selected
    # features) as score. Use to find optimal lambda
    elif question == "feature_select_lambda":
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))

        # hyper-parameters
        lammy_max = 0.05  # max lambda value to try up to (not inclusive). Start from 0
        lammy_interval = 0.0005  # lambda value interval to increase each iteration
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
            print("For lambda = %0.4f, E_train: %0.3f, E_valid: %0.3f, E_approx: %0.3f" % (lammy, E_train, E_valid,
                                                                                           E_valid - E_train))

    # Run more extensive bolasso over given lambda value. Prints features to keep and drop. Saves data set with only
    # significant features (and y or SalePrice) as a csv.
    elif question == "feature_select":
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))

        # hyper-parameters
        lammy = 0.01
        n_bootstraps = 400

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
        data.to_csv('../data/{}_sig_features.csv'.format(dataset), index=False)

        # report
        print("Number of features dropped: ", len(col_drop_list))
        print("Number of features kept: ", len(label_name_list))
        print("These are the features kept: ")
        for feature in label_name_list:
            print(feature)
    
    
    elif question == "anova":
        import warnings
        warnings.filterwarnings("ignore")
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))

        # hyper-parameters
        numfeat = 50

        # Create Validation
        # leave it as None initially to explore the variance first, then change to the choosen number from explained_variance
        X, y, _, _ = standardize_dataset(data, data)
        X_val, X_remain, y_val, y_remain = train_test_split(X, y, test_size=0.50)

        # Create an SelectKBest object to select features with two best ANOVA F-Values
        selector = SelectKBest(f_regression, k=numfeat)

        # Apply the SelectKBest object to the feat. and target
        best_X = selector.fit_transform(X_val, y_val)
        mask = selector.get_support(indices=True)
        new_features = X_val.columns[mask]

        # Use the selected feature from validation set to now filter the X_remaining
        X_remain = X_remain[new_features]
        feature_names = list(X_remain.columns.values)
        print(feature_names)
        X_train, X_test, y_train, y_test = train_test_split(X_remain, y_remain, test_size=0.3)
        my_model = XGBRegressor()
        my_model.fit(X_train, y_train, verbose=False)

        # make predictions
        predictions = my_model.predict(X_test)
        meanabserr = mean_absolute_error(predictions, y_test)
        print(str(numfeat) + ": Mean Absolute Error : " + str(meanabserr))

        # compare with original model with all original features
        original_model = XGBRegressor()
        original_model.fit(X, y, verbose=False)
        predictions = original_model.predict(X)
        meanabserr = mean_absolute_error(predictions, y)
        print("Original: Mean Absolute Error : " + str(meanabserr))

        # save un-standardized data set with only significant features (this has X and y).
        DFYint = pd.DataFrame(np.ones((data.shape[0], 1)), columns=['y-intercept'])
        data = pd.concat([DFYint, data], axis=1)
        SalePriceCol = data['SalePrice']
        data.drop(labels=['SalePrice'], axis=1, inplace=True)
        data = data[feature_names]
        data.insert(0, 'SalePrice', SalePriceCol)
        data.to_csv('../data/{}_anova_features.csv'.format(dataset), index=False)


    elif question == "anovaplot":
        import warnings
        warnings.filterwarnings("ignore")
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))

        # hyper-parameters
        k_features = 244

        abserrs = np.zeros(k_features)
        for numfeat in range(1, k_features+1):
            # Create Validation
            # leave it as None initially to explore the variance first, then change to the choosen number from explained_variance
            X, y, _, _ = standardize_dataset(data, data)
            X_val, X_remain, y_val, y_remain = train_test_split(X, y, test_size=0.50)

            # Create an SelectKBest object to select features with two best ANOVA F-Values
            selector = SelectKBest(f_regression, k=numfeat)

            # Apply the SelectKBest object to the feat. and target
            best_X = selector.fit_transform(X_val, y_val)
            mask = selector.get_support(indices=True)
            new_features = X_val.columns[mask]
            # Use the selected feature from validation set to now filter the X_remaining
            X_remain = X_remain[new_features]
            # print(X_remain.shape)
            feature_names = list(X_remain.columns.values)
            # print(feature_names)
            X_train, X_test, y_train, y_test = train_test_split(X_remain, y_remain, test_size=0.3)
        
            my_model = XGBRegressor()
            my_model.fit(X_train, y_train, verbose=False)

            # make predictions
            predictions = my_model.predict(X_test)
            meanabserr = mean_absolute_error(predictions, y_test)
            print(str(numfeat) + ": Mean Absolute Error : " + str(meanabserr))
            abserrs[numfeat-1] = meanabserr
        
        plt.title("The effect of number of features of ANOVA on testing/training error")
        plt.plot(np.arange(1,numfeat+1), abserrs, label="Abs. error")
        plt.xlabel("# of Features")
        plt.ylabel("Mean Abs. Error")
        plt.legend()

        fname = os.path.join("..", "figs", "ANOVA_validation_plot.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)   

        # compare with original model with all original features
        original_model = XGBRegressor()
        original_model.fit(X, y, verbose=False)

        predictions = original_model.predict(X)
        meanabserr = mean_absolute_error(predictions, y)
        print("Original: Mean Absolute Error : " + str(meanabserr))


    elif question == "xgbregressor":
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))
        X, y, _, _ = standardize_dataset(data, data)
        X_val, X_remain, y_val, y_remain = train_test_split(X, y, test_size=0.50)

        #Rank features on importance using xgbregressor
        xgb = XGBRegressor()
        xgb.fit(X_val, y_val)
        feature_ranking = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_val.columns)
        feature_ranking = feature_ranking.sort_values(['Importance'], ascending = False)
        
        #Hyper parameter test of num features
        numfeat = 50
     
        selected_features = feature_ranking.index.values[1:numfeat]
        print(selected_features)

        #generate new dataset on the remaining data
        X_remain_formodel = X_remain[selected_features]

        # # make predictions using remaining dataset
        X_train, X_test, y_train, y_test = train_test_split(X_remain_formodel, y_remain, test_size=0.50)

        reduced_model = XGBRegressor()
        reduced_model.fit(X_test, y_test, verbose=False)

        predictions = reduced_model.predict(X_test)
        meanabserr = mean_absolute_error(predictions, y_test)
        print(str(numfeat) + ": Selected Features: Mean Absolute Error : " + str(meanabserr))  

        # compare with original model with all original features
        original_model = XGBRegressor()
        original_model.fit(X, y, verbose=False)
        predictions = original_model.predict(X)
        meanabserr = mean_absolute_error(predictions, y)
        print("Original: Mean Absolute Error : " + str(meanabserr))

        # save un-standardized data set with only significant features (this has X and y).
        DFYint = pd.DataFrame(np.ones((data.shape[0], 1)), columns=['y-intercept'])
        data = pd.concat([DFYint, data], axis=1)
        SalePriceCol = data['SalePrice']
        data.drop(labels=['SalePrice'], axis=1, inplace=True)
        data = data[selected_features]
        data.insert(0, 'SalePrice', SalePriceCol)
        data.to_csv('../data/{}_xgb_features.csv'.format(dataset), index=False)


    elif question == "xgbregressorplot":
        data = pd.read_csv('../data/{}_preprocessed.csv'.format(dataset))
        X, y, _, _ = standardize_dataset(data, data)
        X_val, X_remain, y_val, y_remain = train_test_split(X, y, test_size=0.50)

        #Rank features on importance using xgbregressor
        xgb = XGBRegressor()
        xgb.fit(X_val, y_val)
        feature_ranking = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_val.columns)
        feature_ranking = feature_ranking.sort_values(['Importance'], ascending = False)
        
        #Hyper parameter test of num features
        k_features = 244
        abserrs = np.zeros(k_features)
        for numfeat in range(1, k_features+1):        
            selected_features = feature_ranking.index.values[1:numfeat]
            # print(selected_features)

            #generate new dataset on the remaining data
            X_remain_formodel = X_remain[selected_features]

            # # make predictions using remaining dataset
            X_train, X_test, y_train, y_test = train_test_split(X_remain_formodel, y_remain, test_size=0.50)

            reduced_model = XGBRegressor()
            reduced_model.fit(X_test, y_test, verbose=False)

            predictions = reduced_model.predict(X_test)
            meanabserr = mean_absolute_error(predictions, y_test)
            print(str(numfeat) + ": Selected Features: Mean Absolute Error : " + str(meanabserr))
            abserrs[numfeat-1] = meanabserr

        plt.title("The effect of number of features of ANOVA on testing/training error")
        plt.plot(np.arange(1,numfeat+1), abserrs, label="Abs. error")
        plt.xlabel("# of Features")
        plt.ylabel("Mean Abs. Error")
        plt.legend()

        fname = os.path.join("..", "figs", "XGBRegressor_validation_plot.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)   

        # compare with original model with all original features
        original_model = XGBRegressor()
        original_model.fit(X, y, verbose=False)

        predictions = original_model.predict(X)
        meanabserr = mean_absolute_error(predictions, y)
        print("Original: Mean Absolute Error : " + str(meanabserr))


    elif question == "base_models":
        dataset_name = 'preprocessed'
        # dataset_name = 'sig_features'
        # dataset_name = 'anova_features'
        # dataset_name = 'xgb_features'

        # read preprocessed data as pandas dataframe
        df = pd.read_csv('../data/train_{}.csv'.format(dataset_name))
        feats = df.drop('SalePrice', axis=1, inplace=False).columns.values      # features
        X = df.drop('SalePrice', axis=1, inplace=False).values
        y = df['SalePrice'].values

        n, d = np.shape(X)

        err_type = 'rmse'  # 'abs', 'squared', 'rmsle'
        cross_val = True
        valid_size = 0.25
        n_splits = 4
        shuffle_data = True

        ## base models
        # test Lasso model
        print("\nbase model: Lasso (L2-loss with L1-reg)")
        model = Lasso(alpha=0.0005, random_state=1)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                        n_splits=n_splits, verbose=True, err_type=err_type)

        # # doesn't really make a difference
        # print("\nLasso robustscaler")
        # model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
        # err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
        #                                 n_splits=n_splits, verbose=True, err_type=err_type)


        # ElasticNet
        print("\nbase model: ElasticNet (L2-loss with L1-reg and L2-reg)")
        model = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=2)
        # model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))

        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                        n_splits=n_splits, verbose=True, err_type=err_type)
        # Ridge
        print("\nbase model: Ridge (L2-loss with L2_reg)")
        model = Ridge(alpha=1, random_state=2)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)

        # KNN regression
        print("\nbase model: KNN regression")
        model = KNeighborsRegressor(n_neighbors=10)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)

        # lightGBM
        print("\nbase model: lightGBM")
        model = lgb.LGBMRegressor(objective='regression', num_leaves=25, learning_rate=0.05)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)

        # Random forest regressor
        print("\nbase model: Random Forest Regressor")
        model = RandomForestRegressor(n_estimators=100, bootstrap=True, max_depth=12, n_jobs=-1)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)
        
        print("\nbase model: Gradient Boosting")
        model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                    max_depth=12, max_features='sqrt',
                                    min_samples_leaf=4, min_samples_split=10, 
                                    loss='huber', random_state =5)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)
        
        # Neural Net
        print("\nbase model: Neural Net")
        model = NeuralNetRegressor(d, gpu=True, lr=1e-3, momentum=0.9, 
                                    lammy=1e-5, batch_size=32, epochs=100, 
                                    num_workers=6, verbose=False)
        err_tr, err_va = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                        n_splits=n_splits, verbose=True, err_type=err_type)

    elif question == "nn_hyperparams":
        # read preprocessed data as pandas dataframe
        df = pd.read_csv('../data/train_sig_features.csv')
        feats = df.drop('SalePrice', axis=1, inplace=False).columns.values      # features
        X = df.drop('SalePrice', axis=1, inplace=False).values
        y = df['SalePrice'].values

        n, d = np.shape(X)


        ## hyper-params
        # initial value of tunable hyper params
        lr = 1e-3       # learning rate
        momentum = 0.9
        lammy = 1e-5
        batch_size = 32
        # other NN function arguments
        epochs = 100
        num_workers = 6

        # param for tuning process
        num_divs = 10
        err_type = 'squared'
        valid_size = 0.2
        n_splits = 5
        
        # tuning hyper-param: leanring rate
        print(">>>start tuning lr")
        lrs = np.linspace(1e-5, 1e-2, num=num_divs)
        neuralNetHyperparamTuning(
            'lr', lrs, X, y, lr=lr, momentum=momentum, lammy=lammy, batch_size=batch_size, 
            epochs=epochs, num_workers=num_workers, err_type=err_type, 
            cross_val=True, valid_size=valid_size, n_splits=n_splits, save_fig=True)

        # param for momentum tuning
        print(">>>start tuning momentum")
        momentums = np.linspace(0.0, 0.9, num=num_divs)
        neuralNetHyperparamTuning(
            'momentum', momentums, X, y, lr=lr, momentum=momentum, lammy=lammy, batch_size=batch_size, 
            epochs=epochs, num_workers=num_workers, err_type=err_type, 
            cross_val=True, valid_size=valid_size, n_splits=n_splits, save_fig=True)

        # lammy
        print(">>>start tuning lammy")
        lammies = np.linspace(1e-5, 1, num=num_divs)
        neuralNetHyperparamTuning(
            'lammy', lammies, X, y, lr=lr, momentum=momentum, lammy=lammy, batch_size=batch_size, 
            epochs=epochs, num_workers=num_workers, err_type=err_type, 
            cross_val=True, valid_size=valid_size, n_splits=n_splits, save_fig=True)
        
        # batch size
        print(">>>start tuning batch_size")
        # num_divs = 10
        # batch_sizes = np.linspace(4, n, num=num_divs, dtype=int)
        batch_sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, n])
        neuralNetHyperparamTuning(
            'batch_size', batch_sizes, X, y, lr=lr, momentum=momentum, lammy=lammy, batch_size=batch_size, 
            epochs=epochs, num_workers=num_workers, err_type=err_type, 
            cross_val=True, valid_size=valid_size, n_splits=n_splits, save_fig=True)


    elif question == "averaging":
        # dataset_name = 'preprocessed'
        # dataset_name = 'sig_features'
        # dataset_name = 'anova_features'
        dataset_name = 'xgb_features'

        # read preprocessed data as pandas dataframe
        df = pd.read_csv('../data/train_{}.csv'.format(dataset_name))
        feats = df.drop('SalePrice', axis=1, inplace=False).columns.values      # features
        X = df.drop('SalePrice', axis=1, inplace=False).values
        y = df['SalePrice'].values

        n, d = np.shape(X)
        print("num features: {}".format(d))

        err_type = 'rmse'  # 'abs', 'squared', 'rmsle'
        cross_val = True 
        valid_size = 0.25
        n_splits = 4
        shuffle_data = True

        models = []

        models.append(Lasso(alpha=0.5, random_state=None))
        models.append(ElasticNet(alpha=0.5, l1_ratio=0.5))
        models.append(Ridge(alpha=1))
        models.append(KNeighborsRegressor(n_neighbors=10))
        # models.append(GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05))
        models.append(lgb.LGBMRegressor(objective='regression', num_leaves=25, learning_rate=0.05))
        models.append(RandomForestRegressor(n_estimators=100, bootstrap=True, max_depth=12, n_jobs=-1))
        
        # FIXME: NN basemodel, training takes a lot of time
        # models = NeuralNetRegressor(d, gpu=True, lr=1e-3, momentum=0.9, 
        #                             lammy=1e-5, batch_size=32, epochs=100, 
        #                             num_workers=6, verbose=False)

        avg_model = AveragingRegressor(models)
        err_tr, err_va = evaluate_model(avg_model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                        n_splits=n_splits, verbose=True, err_type=err_type)

    elif question == "stacking":
        dataset_name = 'preprocessed'
        # dataset_name = 'sig_features'
        # dataset_name = 'anova_features'
        # dataset_name = 'xgb_features'

        # read preprocessed data as pandas dataframe
        df = pd.read_csv('../data/train_{}.csv'.format(dataset_name))
        feats = df.drop('SalePrice', axis=1, inplace=False).columns.values      # features
        X = df.drop('SalePrice', axis=1, inplace=False).values
        y = df['SalePrice'].values

        n, d = np.shape(X)
        print("num features: {}".format(d))

        err_type = 'rmse'  # 'abs', 'squared', 'rmsle'
        cross_val = True 
        valid_size = 0.25
        n_splits = 4
        shuffle_data = True

        base_models = []
        base_models.append(Lasso(alpha=0.0005, random_state=None))
        base_models.append(ElasticNet(alpha=0.0005, l1_ratio=0.9))
        base_models.append(Ridge(alpha=1))
        base_models.append(KNeighborsRegressor(n_neighbors=10))
        base_models.append(lgb.LGBMRegressor(objective='regression', num_leaves=25, learning_rate=0.05))
        base_models.append(RandomForestRegressor(n_estimators=100, bootstrap=True, max_depth=12, n_jobs=-1))

        # gradient boosting takes way too much time (even longer than NN)
        # base_models.append(GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05))
        
        # FIXME: NN basemodel, training takes a lot of time
        # base_models = NeuralNetRegressor(d, gpu=True, lr=1e-3, momentum=0.9, 
        #                             lammy=1e-5, batch_size=32, epochs=100, 
        #                             num_workers=6, verbose=False)

        meta_model = Lasso(alpha=1e-4) 
        # meta_model = ElasticNet(alpha=1, l1_ratio=0.5)
        # meta_model = Ridge(alpha=1)
        
        stacking_model = StackingRegressor(base_models, meta_model, n_folds=5)
        err_tr, err_va = evaluate_model(stacking_model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                        n_splits=n_splits, verbose=True, err_type=err_type)

        print("Kaggle implementation")
        stacking_model = StackingAveragedModels(base_models, meta_model, n_folds=5)
        err_tr, err_va = evaluate_model(stacking_model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                        n_splits=n_splits, verbose=True, err_type=err_type)


    elif question == "base_model_tuning":
        # dataset_name = 'sig_features'
        dataset_name = 'preprocessed'
        # dataset_name = 'anova_features'
        
        df = pd.read_csv('../data/train_{}.csv'.format(dataset_name))
        feats = df.drop('SalePrice', axis=1, inplace=False).columns.values      # features
        X = df.drop('SalePrice', axis=1, inplace=False).values
        y = df['SalePrice'].values

        cross_val = False
        valid_size = 0.25
        n_splits = 4
        err_type = 'squared'
        shuffle_data = False

        num_vars = 50

        #
        errs_col_name = ['Lasso', 'ElasticNet', 'Ridge', 'KNNRegressor', 'lightGBM', 'RandomForestRegressor']
        var_name = ['alpha', 'alpha', 'alpha', 'n_neighbors', 'num_leaves', 'max_depth']
        
        # tunable vars
        # alphas = np.linspace(0.1, 5, num=num_vars)
        alphas = np.linspace(1e-4, 1, num=num_vars)
        n_neighbors = np.linspace(1, 50, num=num_vars, dtype=int)
        num_leaves = np.linspace(5, 200, num=num_vars, dtype=int)
        max_depths = np.linspace(5, 100, num=num_vars, dtype=int)

        errs_tr = np.empty((len(alphas), len(errs_col_name)))    # col0: Lasso, col1: ENet, col2: Ridge
        errs_va = np.empty((len(alphas), len(errs_col_name)))

        for i in range(num_vars):
            
            # Lasso
            model = Lasso(alpha=alphas[i])
            errs_tr[i][0], errs_va[i][0] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data, 
                                                            n_splits=n_splits, verbose=True, err_type=err_type)

            # ElasticNet
            model = ElasticNet(alpha=alphas[i], l1_ratio=0.2)
            errs_tr[i][1], errs_va[i][1] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                                            n_splits=n_splits, verbose=True, err_type=err_type)

            # Ridge
            model = Ridge(alpha=alphas[i])
            errs_tr[i][2], errs_va[i][2] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                                            n_splits=n_splits, verbose=True, err_type=err_type)
            
            # KNN regressor
            model = KNeighborsRegressor(n_neighbors=n_neighbors[i], n_jobs=-1)
            errs_tr[i,3], errs_va[i,3] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                                n_splits=n_splits, verbose=True, err_type=err_type)

            # light GBM
            model = lgb.LGBMRegressor(objective='regression', num_leaves=num_leaves[i], learning_rate=0.05)
            errs_tr[i,4], errs_va[i,4] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                                n_splits=n_splits, verbose=True, err_type=err_type)

            # random forest 
            model = RandomForestRegressor(n_estimators=100, max_depth=max_depths[i], bootstrap=True, n_jobs=-1)
            errs_tr[i,5], errs_va[i,5] = evaluate_model(model, X, y, cross_val=cross_val, valid_size=valid_size, shuffle_data=shuffle_data,  
                                                n_splits=n_splits, verbose=True, err_type=err_type)


        vars = np.column_stack([alphas, alphas, alphas, n_neighbors, num_leaves, max_depths])

        for j, name in enumerate(errs_col_name, 0):
            plt.figure()
            plt.plot(vars[:,j], errs_tr[:,j], label="training errors")
            plt.plot(vars[:,j], errs_va[:,j], label="validation errors")
            plt.xlabel('{}'.format(var_name[j]))
            plt.ylabel('errors ({})'.format(err_type))
            plt.title('{} tuning (cross_validation: {})'.format(name, str(cross_val)))
            plt.legend()
            plt.grid()
            fname=os.path.join('..','figs','[{}]-{}_tuning.png'.format(dataset_name, name))
            plt.savefig(fname)

        plt.show()
