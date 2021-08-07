import streamlit as st
from sklearn.neural_network import MLPRegressor

def mlpreg_param_selector():
    
    solver = st.selectbox("solver", ('adam','sgd', 'lbfgs'))
    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)
    learning_rate = st.selectbox("learning_rate", ('constant','invscaling','adaptive'))
    params = {"solver": solver, "max_iter": max_iter, "learning_rate": learning_rate}
    model = MLPRegressor(**params)
    return model
