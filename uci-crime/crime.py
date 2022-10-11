# DATA SLAYERS SLAYERS

import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.feature_selection import RFE, SelectFromModel, f_regression

# print("Your version of SKlearn is ", sklearn.__version__, "if it is not 1 you need to update")

# FOUR ASSUMPTIONS OF LINEAR REGRESSION
# LINE
# Linearity
# Independence
# Normality
# Equal variance


def get_data():
    # region IMPORT AND CLEANING
    # raw = read_csv('/home/thomaswit/DataSCapstone/capstone-case-studies/uci-crime/crimedata.csv')
    # print(raw.head(5))
    # print(raw.shape)
    raw = read_csv('crimedata.csv')
    # print(raw.head(5))
    # print("base", raw.shape)

    data = raw
    data.replace('?', np.NaN, inplace=True)
    # Drop because not relevant
    data.drop(['communityname', 'state', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)
    # Drop because TOO relevant to crime prediction
    data.drop(data.columns[124:-1], axis=1, inplace=True)
    # Drop because can't make judgement off this
    data.drop(
        ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "whitePerCap", "blackPerCap", "indianPerCap",
         "AsianPerCap", "OtherPerCap", "HispPerCap", ], axis=1, inplace=True)
    # print("cut crime:", data.shape)
    # print(data.columns)
    # Dropping columns with mostly missing data
    na_vec = data.isnull().sum()
    count = 0
    to_drop = []
    for x in na_vec:
        if x / data.shape[0] > .05:
            to_drop.append(count)
        count += 1
    data.drop(data.columns[to_drop], axis=1, inplace=True)
    # print("remove columns with many ?:", data.shape)
    # Drop any rows with missing values
    data.dropna(inplace=True)
    # print("remove rows with ?:", data.shape)
    # data = data.apply(pd.to_numeric)
    # Dropping heavily missing columns before dropping rows leaves 2117 x 121
    # Dropping just rows leaves us with 302 x 142
    return data

def main():
    data = get_data()

    y_name = "nonViolPerPop"
    data = remove_outliers(data, y_name)

    # Produce response vector
    y = data[y_name].tolist()
    X = data.drop(y_name, axis=1)

    # scale
    X, y = power_transform(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.columns)

    # feature selection
    X_train, X_test = select_features(X_train, y_train, X_test, 'all')
    # all, or val for number of features wanted

    lin = lin_reg(X_train, y_train, X_test, y_test)
    lasso = lasso_reg(X_train, y_train, X_test, y_test)
    resid_df = calculate_residuals(lasso, X_train, y_train)
    outlier_indices = resid_df[resid_df['Residuals'] >= 3 * np.std(resid_df['Residuals'])].index
    print('swag')

    lasso2 = linear_model.Lasso(alpha=10)
    lasso2.fit(X_train, y_train)
    print(len(lasso.coef_))
    print(lasso.coef_)
    print(lasso.intercept_)
    # print(lasso.feature_names_in_)

    # SHOW DATA
    # sns.pairplot(data)
    # plt.show()
    # data.hist()
    # plt.show()
    # data.plot()
    # plt.show()


# region preprocessing
def remove_outliers(data, y_name):
    z = np.abs(stats.zscore(data[y_name]))
    d_c = data[(z<3)]
    print('REMOVING OUTLIERS of nonViol, 3 std_devs')
    print(data.shape, d_c.shape)
    return d_c


def power_transform(X, y):
    pt = preprocessing.PowerTransformer()
    X = pt.fit_transform(X)
    # y = pt.fit_transform(np.array(y).reshape(-1,1))
    return X, y


def select_features(X_train, y_train, X_test, k_val):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k=k_val)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    # print(fs.feature_names_in_)
    # Show Feature Scores
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    # plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # plt.show()
    return X_train_fs, X_test_fs


def correlation_matrix(data):
    sns.set(rc={"figure.figsize": (90, 90)})
    corr_mat = data.corr().round(3)
    # mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    # sns.heatmap(corr_mat, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask).figure.savefig("heat_map_2.png")
    corr_mat = corr_mat.unstack()
    high_corr = corr_mat[abs(corr_mat) > 0.7]
    high_corr = high_corr[1 > high_corr]
    print("HIGH CORRELATION")
    print(high_corr)
    # f = open("high_corr.txt", x)
    # f.write(high_corr)
    # f.close()
    # plt.show()


# Correlation Matrix
# correlation_matrix(data)



# endregion


# region Regression


def lin_reg(X_train, y_train, X_test, y_test):
    print("###### Linear Regression #####")
    reg = LinearRegression().fit(X_train, y_train)
    print("R^2: ", reg.score(X_test, y_test))
    # print("coef: ", reg.coef_)
    # mod = sm.OLS(y_train,X_train_fs)
    # fii = mod.fit()
    # p_values = fii.summary2().tables[1]['P>|t|']
    # print(p_values[p_values < .05])
    return reg


def lin_reg_summary(X_train, y_train):
    # Linear Regression Summary
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=50, step=25)
    selector = selector.fit(X_train, y_train)


def lasso_reg(X_train, y_train, X_test, y_test):
    print("##### Lasso #####")
    lasso = linear_model.Lasso(alpha=.1, max_iter=1000)
    # # lasso = linear_model.LassoLarsIC(criterion="aic", fit_intercept=True, max_iter=100000)
    lasso.fit(X_train, y_train)
    pred_train_lasso = lasso.predict(X_train)

    print("R2_train: ", r2_score(y_train, pred_train_lasso))
    # print(np.sqrt(mean_squared_error(y_train, pred_train_lasso)))
    print(len(lasso.coef_))

    pred_test_lasso = lasso.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, pred_test_lasso)))
    print(r2_score(y_test, pred_test_lasso))
    print("Score: ", lasso.score(X_test, y_test))
    # # print("Alpha: ", lasso.alpha)
    # print("Coeficients:", lasso.coef_)
    # print("intercepts:", lasso.intercept_)
    # print("features:", lasso.n_features_in_)
    # print(lasso.feature_names_in_)
    return lasso


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results


# code to check assumptions
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')

    print('Checking with a scatter plot of actual vs. predicted.',
          'Predictions should follow the diagonal line.')

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False)

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()


# linear_assumption(lasso, X_train, y_train)


def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.

    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')

    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)

    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)

    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')

    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()

    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')


# normal_errors_assumption(lasso, X_train, np.log(y_train))

# endregion


if __name__ == "__main__":
    main()
