# DATA SLAYERS SLAYERS

import pandas as pd
from scipy import stats
from seaborn import lmplot
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
from sklearn.preprocessing import MinMaxScaler

# print("Your version of SKlearn is ", sklearn.__version__, "if it is not 1 you need to update")

# FOUR ASSUMPTIONS OF LINEAR REGRESSION
# LINE
# Linearity
# Independence
# Normality
# Equal variance

"""
Data Cleaning:
We begun by removing the first 5 columns, as these were labels and not useful features. Then we removed all columns
that could be considered response variables, or contributed to our primary response: nonViolPerPop. We also decided
to remove the columns that indicated the racial breakdown of the communities, as it would not be ethical to consider
these as explanatory variables for crime.
Next, we needed to consider how to deal with missing data, which this dataset had plenty of. If we decided to simply
drop all rows with missing data, our data goes from 2215 samples to 302. However, we noticed that there were some
columns that had mostly missing data. So, we decided to drop any columns that were missing over 80% of their
values. Surprisingly, almost all of the missing values were accounted for in the columns we dropped. After now dropping
any rows with missing values, we were left with our clean data: 2118 data points with 92 explanatory variables. We found
that nonViolPerPop (our response variable) was missing about 90 values. Because these rows must be dropped (we can't
use them for creating a model) most of the lost samples were due to this column and couldn't be avoided. Our data went
from 2215 to 2118 samples, around 90 of the rows dropped being necessary. We believe this was the best way to keep as
much relevant data as possible, both in rows and columns.
"""


def get_data():
    # region IMPORT AND CLEANING
    # raw = read_csv('/home/thomaswit/DataSCapstone/capstone-case-studies/uci-crime/crimedata.csv')
    # print(raw.head(5))
    raw = read_csv('crimedata.csv')
    # print(raw.shape)
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
        if x / data.shape[0] > .80:
            to_drop.append(count)
        count += 1
    data.drop(data.columns[to_drop], axis=1, inplace=True)
    # print("remove columns with many ?:", data.shape)
    # Drop any rows with missing values
    data.dropna(inplace=True)
    # print("remove rows with ?:", data.shape)
    data = data.apply(pd.to_numeric)
    # Dropping heavily missing columns before dropping rows leaves 2118 x 93
    # Dropping just rows leaves us with 302 x 142
    return data


def main():
    data = get_data()

    y_name = "nonViolPerPop"
    data = remove_outliers(data, y_name)

    # Correlation Matrix
    # correlation_matrix(data)

    # Produce response vector
    y = data[y_name].tolist()
    X = data.drop(y_name, axis=1)

    # scale
    X, y = scale(X, y)
    X, y = power_transform(X, y)
    y = list(y.T[0])

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

    # lasso2 = linear_model.Lasso(alpha=10)
    # lasso2.fit(X_train, y_train)
    # print(len(lasso.coef_))
    # print(lasso.coef_)
    # print(lasso.intercept_)
    # print(lasso.feature_names_in_)

    # lin = lin_reg(X_train, y_train, X_test, y_test)
    # lasso = lasso_reg(X_train, y_train, X_test, y_test)
    ridge_ = ridge_reg(X_train, y_train, X_test, y_test)

    # print('for debugging')
    linear_assumption(lasso, X_train, y_train)
    linear_assumption(ridge_, X_train, y_train)
    #
    # normal_errors_assumption(lasso, X_train, np.log(y_train))
    # normal_errors_assumption(ridge_, X_train, np.log(y_train))

    # SHOW DATA
    # sns.pairplot(data)
    # plt.show()
    # data.hist()
    # plt.show()
    # data.plot()
    # plt.show()

# endregion

"""
Preprocessing:
In preprocessing our data, we kept in mind our 4 LINE assumptions. First, we scaled the data down, to improve usability,
readability, and prevent data size from being an issue. Second, we removed some outliers from the data. We removed any city
who's nonViolPerPop was over 3 standard deviations out. With this we removed a total of 24 cities, including
Tukwilacity WA, which was 7 standard deviations above the median, and EastLongmeadowtown MA, 8 standard deviations above.
From there we challenged the assumptions of linearity and normality by running it through a power transformation. As there
were 0's in the dataset, we couldn't use the Box-cox method, and so used the Yeo-johnson algorithm. From research, it seems
that yeo-johnson is more versatile and just as accurate, though it makes it harder to understand exactly what the relationship
is between the data. From here we challenged the assumption of independence by creating and examining a correlation
matrix for the data, which allowed us to see to what extent the data is related to itself and thus lead to potential
mistakes through co-linearity. As the matrix showed a lot of correlation, we knew we would need to employ some form of
feature selection, and so initially implemented a SelectKBest algorithm which chose an amount of variables based on their
F-statistic. We later found this to be insufficient, and had better luck running a Lasso/Ridge reduction method. 
"""

# region preprocessing


def scale(X, y):
    X = MinMaxScaler().fit_transform(X)
    y = MinMaxScaler().fit_transform(np.array(y).reshape(-1,1))
    return X, y


def remove_outliers(data, y_name):
    z = np.abs(stats.zscore(data[y_name]))
    d_c = data[(z<3)]
    print('REMOVING OUTLIERS of nonViol, 3 std_devs')
    print(data.shape, d_c.shape)
    return d_c


def power_transform(X, y):
    pt = preprocessing.PowerTransformer()
    X = pt.fit_transform(X)
    y = pt.fit_transform(np.array(y).reshape(-1,1))
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
    lasso = linear_model.Lasso(alpha=.02, max_iter=13000)
    # # lasso = linear_model.LassoLarsIC(criterion="aic", fit_intercept=True, max_iter=100000)
    lasso.fit(X_train, y_train)
    pred_train_lasso = lasso.predict(X_train)

    print("R2 Score on Train: " , r2_score(y_train, pred_train_lasso))
    # print(np.sqrt(mean_squared_error(y_train, pred_train_lasso)))

    pred_test_lasso = lasso.predict(X_test)
    # print(np.sqrt(mean_squared_error(y_test, pred_test_lasso)))
    # print(r2_score(y_test, pred_test_lasso))
    print("R2 Score on test: ", lasso.score(X_test, y_test))
    # # print("Alpha: ", lasso.alpha)
    print("Coeficients:", lasso.coef_)
    # print("intercepts:", lasso.intercept_)
    # print("features:", lasso.n_features_in_)
    # print(lasso.feature_names_in_)
    return lasso


def ridge_reg(X_train, y_train, X_test, y_test):
    print('##### Ridge #####')
    ridge = linear_model.Ridge(alpha=0.01, max_iter=13000)
    ridge.fit(X_train, y_train)
    print(ridge.coef_)
    pred_train_ridge = ridge.predict(X_train)

    print(np.sqrt(mean_squared_error(y_train, pred_train_ridge)))
    print("R2 train: ", r2_score(y_train, pred_train_ridge))
    # print(len(ridge.coef_))

    pred_train_ridge = ridge.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, pred_train_ridge)))
    # print(r2_score(y_test, pred_train_ridge))
    print("R2 Test Score: ", ridge.score(X_test, y_test))
    return ridge


lin = lin_reg(X_train, y_train, X_test, y_test)
lasso = lasso_reg(X_train, y_train, X_test, y_test)
ridge_ = ridge_reg(X_train, y_train, X_test, y_test)
print('for debugging')


def get_nonzero_features(which_model, which_dataset):
    """Create a dictionary of all the nonzero coefficients from a regression model.

    :param which_model: a sklearn model
    :param which_dataset: the dataset that the model is derived from. The column names must be included!
    :return: a dictionary, with the feature names as keys and the nonzero coefficients as values.
    """

    model_coefficients = pd.DataFrame(which_model.coef_)
    data_cols = which_dataset.columns
    list_of_nonzero_feature_names = []
    nonzero_feature_coefficients = []

    for i in range(0, model_coefficients.shape[0]):
        if float(model_coefficients.iloc[i, 0]) != float(0):
            nonzero_feature_coefficients.append(model_coefficients.iloc[i, 0])
            list_of_nonzero_feature_names.append(data_cols[i])

    nonzero_feature_dict = {}

    for key, value in zip(list_of_nonzero_feature_names, nonzero_feature_coefficients):
        nonzero_feature_dict[key] = value

    return nonzero_feature_dict


lasso_feature_name_dict = get_nonzero_features(lasso, data)
ridge_feature_name_dict = get_nonzero_features(ridge_, data)


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results


# code to check assumptions
# if the plots aren't showing up in Pycharm, go to file>settings>tools>python scientific>show plots in tool window
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


# endregion


if __name__ == "__main__":
    main()
