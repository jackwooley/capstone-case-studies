import crime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Train/test split function
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
Here is where we attempted to create a decision tree model. We used the same cleaned data from before. 2118 x 92.
Its worth noting that sklearn has different versions of the decision tree package. For this model, we needed the
regressor decision tree, as our response variable was continuous.
"""

data = crime.get_data()
y_name = "nonViolPerPop"

# Produce response vector
y = data[y_name]
X = data.drop(y_name, axis=1)
feature_cols = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree regressor object
dt = DecisionTreeRegressor()

# Train Decision Tree
dt = dt.fit(X_train, y_train)

# Predict the response for train dataset
y_pred_train = dt.predict(X_train)

print("R2 on Train: ", r2_score(y_train, y_pred_train))
print("RMSE on Train: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))

# Predict the response for test dataset
y_pred = dt.predict(X_test)

print("R2 on Test: ", r2_score(y_test, y_pred))
print("RMSE on Test: ", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.plot(y_test.tolist(), 'r.', y_pred.tolist(), 'b.')
plt.show()


# Display the Decision Tree
# Note - Value shows a list that is the count of each class at that point in the tree
# def displayTree(tree):
#     dot_data = StringIO()
#     export_graphviz(tree, out_file=dot_data,
#                     filled=True, rounded=True, impurity=True,
#                     special_characters=True, feature_names=feature_cols)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('crime_tree.png')
#     Image(graph.create_png())
#
#
# displayTree(clf)
# Image('crime_tree.png')
