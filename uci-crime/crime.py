# AUSTIN BAILEY

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

raw = read_csv('crimedata.csv')
print(raw.head(5))
print(raw.shape)

# FOUR ASSUMPTIONS OF LINEAR REGRESSION
# LINE
# Linearity
# Independence
# Normality
# Equal variance

# DATA CLEANING
data = raw
y = "nonViolPerPop"
data.replace('?', np.NaN, inplace=True)
# Drop because not relevant
data.drop(['communityname', 'state', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)
# Drop because TOO relevant to crime prediction
data.drop(data.columns[124:-1], axis=1, inplace=True)
print(data.shape)
print(data.columns)
# Dropping columns with mostly missing data
na_vec = data.isnull().sum()
count = 0
to_drop = []
for x in na_vec:
    if x / data.shape[0] > .10:
        to_drop.append(count)
    count += 1
data.drop(data.columns[to_drop], axis=1, inplace=True)
print(data.shape)
# Drop any rows with missing values
data.dropna(inplace=True)
print(data.shape)
data = data.apply(pd.to_numeric)
# Dropping heavily missing columns before dropping rows leaves 2117 x 121
# Dropping just rows leaves us with 302 x 142

# Produce response vector
response = data[y].tolist()
data = data.drop(y, axis=1)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, response, test_size=0.1, random_state=42)

# Run Linear Regression on Data
# reg = LinearRegression().fit(X_train, y_train)
# score = reg.score(X_test, y_test)
# coef = reg.coef_
# int = reg.intercept_
# print("R^2: ", score)
# print("coef: ", coef)
# print("Intercept: ", int)

# Run LASSO on Data
lasso = linear_model.Lasso(alpha=10)
lasso.fit(X_train, y_train)
print(len(lasso.coef_))
print(lasso.coef_)
print(lasso.intercept_)
print(lasso.feature_names_in_)

# Linear Regression Summary
# X2 = sm.add_constant(X_train)
# est = sm.OLS(y_train, X2)
# est2 = est.fit()
# print(est2.summary())
#
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=50, step=25)
# selector = selector.fit(X_train, y_train)


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results
resid_df = calculate_residuals(lasso, X_train, y_train)
outlier_indices = resid_df[resid_df['Residuals'] >= 3 * np.std(resid_df['Residuals'])].index
print('swag')

lasso2 = linear_model.Lasso(alpha=10)
lasso2.fit(X_train, y_train)
print(len(lasso.coef_))
print(lasso.coef_)
print(lasso.intercept_)
print(lasso.feature_names_in_)



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


linear_assumption(lasso, X_train, y_train)


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


normal_errors_assumption(lasso, X_train, np.log(y_train))
