import streamlit as st
from sklearn.linear_model import SGDClassifier


def sgd_param_selector():
    loss = st.selectbox("loss", ("hinge", "log", "modified_huber", "squared_hinge",'perceptron','squared_loss','huber','epsilon_insensitive', 'squared_epsilon_insensitive'))
    penalty = st.selectbox("penalty", ('l2','l1'))
    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)
    params = {"loss": loss, "penalty": penalty, "max_iter": max_iter}
    model = SGDClassifier(**params)
    return model
