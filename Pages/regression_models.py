import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from Models.RandomForestRegressor import rfreg_param_selector
from Models.KNeighborsRegressor import knreg_param_selector
from Models.DecisionTreeRegressor import dtreg_param_selector
from Models.GradientBoostingRegressor import gbreg_param_selector
from Models.MLPRegressor import mlpreg_param_selector
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px


def Regression_X_Y_Selector(dataset):
    st.subheader('Please Select X features and Y target column.')
    X_cols = st.multiselect('X',
                            dataset.columns.tolist(),
                            dataset.columns.tolist(), key='X_cols')


    Y_cols = st.multiselect('Y',
                            dataset.columns.tolist(),
                            key='Y_cols')
    X = dataset[X_cols]
    Y = dataset[Y_cols]
    return X, Y


def Regression_algos_selector():
    model_type = st.sidebar.selectbox(
        'Select Model',
        (
            'RandomForestRegressor',
            'KNeighborsRegressor',
            'DecisionTreeRegressor',
            'GradientBoostingRegressor',
            'MLPRegressor')
    )
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        if model_type == 'RandomForestRegressor':
            model = rfreg_param_selector()
        elif model_type == 'KNeighborsRegressor':
            model = knreg_param_selector()
        elif model_type == 'DecisionTreeRegressor':
            model = dtreg_param_selector()
        elif model_type == 'GradientBoostingRegressor':
            model = gbreg_param_selector()
        elif model_type == 'MLPRegressor':
            model = mlpreg_param_selector()
    return model, model_type


def generate_data_regression(X, Y):
    # split into train and test sets
    size = st.sidebar.slider("Percentage of dataset division",
                             min_value=0.1,
                             max_value=0.9,
                             step=0.1,
                             value=0.3,
                             help="This is the value which will be used to divide the data for training and testing. Default = 30%")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, X_test, y_train, y_test


def main(model, model_type, X_train, X_test, y_train, y_test):
    if 'regression_mse' not in st.session_state:
        st.session_state.regression_mse = pd.DataFrame(columns=['Algorithm', 'Mean_Squared_Error'])
    m_s_e = st.session_state.regression_mse
    model.fit(X_train, y_train.values.ravel())
    pre = model.predict(X_test)
    rms = r2_score(y_test, pre)
    st.subheader(model_type + ' results')
    st.write("regression score function: ", rms)
    rms = mean_squared_error(y_test, pre)
    st.write("Mean Squared Error: ", rms)
    # Figure
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.plot(y_test.values)
    plt.plot(pre)
    plt.title(model_type + 'Prediction')
    plt.ylabel("Target")
    plt.xlabel("Test data points")
    plt.legend(['Actual Values', 'Predicted Values'], loc='upper left')
    st.pyplot(fig)

    bool_var = 0
    for index, row in m_s_e.iterrows():
        if m_s_e.loc[index, 'Algorithm'] == str(model_type):
            m_s_e.loc[index, 'Mean_Squared_Error'] = rms
            bool_var = 1
    if bool_var == 0:
        m_s_e = m_s_e.append({'Algorithm': model_type, 'Mean_Squared_Error': rms}, ignore_index=True)
    m_s_e = m_s_e.sort_values(['Mean_Squared_Error'])
    fig2 = px.bar(m_s_e, x='Algorithm', y='Mean_Squared_Error', title="Mean Squared Error of each Regressor",
                  color='Mean_Squared_Error')
    fig2.update_traces(textposition='outside')
    st.session_state.regression_mse = m_s_e
    st.plotly_chart(fig2)


# @st.cache
def app():
    if 'data' not in st.session_state:
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = st.session_state.data
        st.markdown('# Regression Models')
        options = st.sidebar.selectbox(
            'Options',
            (
                'Select Features and Target',
                'Models',
            )
        )
        if options == 'Select Features and Target':
            st.session_state.X_r, st.session_state.Y_r = Regression_X_Y_Selector(df)
        if options == 'Models':
            try:
                X = st.session_state.X_r
                Y = st.session_state.Y_r
            except:
                st.markdown("Please Select the X and Y through `Select Features and Target` Option!")
                return
            model, model_type = Regression_algos_selector()
            X_train, X_test, y_train, y_test = generate_data_regression(X, Y)
            main(model, model_type, X_train, X_test, y_train, y_test)
