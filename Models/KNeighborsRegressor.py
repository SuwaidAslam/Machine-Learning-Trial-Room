import streamlit as st
from sklearn.neighbors import KNeighborsRegressor


def knreg_param_selector():
    n_neighbors = st.number_input("n_neighbors", min_value=1, max_value=100, step=1)
    weights = st.selectbox("weights", ('uniform','distance'))
    algorithm = st.selectbox("algorithm", ('auto','ball_tree','kd_tree','brute'))
    params = {"n_neighbors": n_neighbors, "weights": weights, "algorithm": algorithm}
    model = KNeighborsRegressor(**params)
    return model
