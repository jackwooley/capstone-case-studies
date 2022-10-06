# AUSTIN BAILEY
# THIS IS A TEST

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import statsmodels.api as sm 
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

raw = read_csv('/home/thomaswit/DataSCapstone/capstone-case-studies/uci-crime/crimedata.csv')
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
lasso = linear_model.Lasso(alpha=.1)
lasso.fit(X_train, y_train)
pred_train_lasso= lasso.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))
print(len(lasso.coef_))

pred_test_lasso= lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))

# print(lasso.coef_)
# print(lasso.intercept_)
# print(lasso.feature_names_in_)
# Linear Regression Summary
# X2 = sm.add_constant(X_train)
# est = sm.OLS(y_train, X2)
# est2 = est.fit()
# print(est2.summary())
#
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=50, step=25)
# selector = selector.fit(X_train, y_train)
