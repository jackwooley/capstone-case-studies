import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("fetal_health.csv")
print(df.head(5))
# (2126, 22)
print(df.shape)
na_vec = df.isnull().sum()
# NO MISSING VALUES
print(na_vec)
# MOSTLY NORMAL
print(df["fetal_health"].value_counts())

y = df["fetal_health"]
X = df.drop(["fetal_health"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# hidden_layer_sizes=(100)
# NOT SCALED: ACCURACY = 0.876
# SCALED: ACCURACY = 0.913

# hidden_layer_size=(100,100)
# ACCURACY = .930

# hidden_layer_size=(100,10,100)
# NOT SCALED: ACCURACY = 0.812
# SCALED: ACCURACY = 0.913

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,100), random_state=1, max_iter=20000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

# GRID SEARCH
# mlp = MLPClassifier(max_iter=13000)
# parameters = {
#     'hidden_layer_sizes': [(50,),(150,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [1e-5, 0.05],
#     'learning_rate': ['constant', 'adaptive'],
# }
#
# clf = GridSearchCV(mlp, parameters, n_jobs=-1, cv=2)
# clf.fit(X, y)
# print("Best parameters found:\n", clf.best_params_)
# y_pred = clf.predict(X_test)
#
# print(accuracy_score(y_test, y_pred))
