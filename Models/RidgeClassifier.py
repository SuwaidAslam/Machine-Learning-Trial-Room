import streamlit as st
from sklearn.linear_model import RidgeClassifier



def rc_param_selector():
    alpha = st.number_input("alpha", min_value=0.0001, max_value=10.0, step=1.0)
    params = {"alpha": alpha}
    model = RidgeClassifier(**params)
    return model
