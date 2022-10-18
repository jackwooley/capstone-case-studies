import mushroom
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

df = mushroom.get_data()
df = mushroom.data_to_dummy(df)

y = df['edible']
X = df.drop(["edible"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

# print(y_test.iloc[0])
# y_test[0] == 1 (TRUE)
# Purposely setting one of the responses to opposite label to check accuracy of 1
# y_test.iloc[0] = 0
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
# ACCURACY OF 1??
