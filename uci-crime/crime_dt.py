import crime
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Train/test split function
from sklearn import metrics  # module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# FIXME
# WARNING: THIS CODE DOES NOT WORK (YET)

data = crime.get_data()
y_name = "nonViolPerPop"

# Produce response vector
y = data.nonViolPerPop
X = data.drop(y_name, axis=1)
feature_cols = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree classifier object - uses gini by default
clf = DecisionTreeClassifier(max_depth=5)

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Display the Decision Tree
# Note - Value shows a list that is the count of each class at that point in the tree
def displayTree(tree):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,
                    filled=True, rounded=True, impurity=True,
                    special_characters=True, feature_names=feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('crime_tree.png')
    Image(graph.create_png())


displayTree(clf)
Image('crime_tree.png')
