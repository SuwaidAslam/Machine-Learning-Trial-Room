import streamlit as st
from sklearn.tree import DecisionTreeRegressor


def dtreg_param_selector():

    criterion = st.selectbox("criterion", ["mse", "friedman_mse", 'mae', 'poisson'])
    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 2, 20, 2)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeRegressor(**params)
    return model
