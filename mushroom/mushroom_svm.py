import mushroom
from sklearn.model_selection import train_test_split
from sklearn import svm

df = mushroom.get_data()
y = df['edible']
X = df.drop(["edible"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X, y)


