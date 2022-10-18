import mushroom
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df = mushroom.get_data()
df = mushroom.data_to_dummy(df)
df.columns = df.columns.astype(str)

y = df['edible']
X = df.drop(["edible"], axis=1)
# Just using whether gill size is broad or not
# Gets accuracy of around 75%!
X = X['b'].iloc[:, 2]
X = np.array(X)
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

# CROSS VAL
scores = cross_val_score(clf, X_train, y_train, cv=10)
print(scores)

# print(y_test.iloc[0])
# y_test[0] == 1 (TRUE)
# Purposely setting one of the responses to opposite label to check accuracy of 1
# y_test.iloc[0] = 0
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
# ACCURACY OF 1??