import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import fetal_functions as f_fn
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st






def main():
    fetal = f_fn.getData()
    X, y = f_fn.split_f_t(fetal, 'fetal_health')
    X_train, X_test, y_train, y_test = f_fn.trainTest(X, y, 0.2, 42)

    st.write(X, y)
    # preprocessing and EDA




    # A few Algorithms
    # knn_preds = f_fn.knn_model(X_train, X_test, y_train, y_test)
    # dt_preds = f_fn.dt_model(X_train, X_test, y_train, y_test)  # TODO: PREVENT OVER-FITTING
    # mlp_preds = f_fn.mlp_model(X_train, X_test, y_train, y_test)
    #
    # f_fn.confusionMatrix(y_test, knn_preds)
    # f_fn.confusionMatrix(y_test, dt_preds)
    # f_fn.confusionMatrix(y_test, mlp_preds)


    # Bagging, boosting
    # voted_preds = f_fn.ensemble(mlp_preds, knn_preds, dt_preds)
    # print("Voting accuracy_score: " + str(accuracy_score(y_test, voted_preds)))
    #
    # f_fn.confusionMatrix(y_test, voted_preds)


    return 1


if __name__ == '__main__':
    main()

