import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from Models.SVC import svc_param_selector
from Models.LogisticRegression import lr_param_selector
from Models.DecisionTree import dt_param_selector
from Models.RandomForet import rf_param_selector
from Models.RidgeClassifier import rc_param_selector
from Models.SGDClassifier import sgd_param_selector
from Models.KNeighborsClassifier import kn_param_selector
from Models.MLPClassifier import mlp_param_selector
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def Classification_X_Y_Selector(dataset):
    st.subheader('Please Select X features and Y target column.')
    X_cols = st.multiselect('X',
                            dataset.columns.tolist(),
                            dataset.columns.tolist(), key='X_cols')

    Y_cols = st.multiselect('Y',
                            dataset.columns.tolist(),
                            key='Y_cols')
    X = dataset[X_cols]
    Y = dataset[Y_cols]
    return X,Y

def Classification_algos_selector():
    model_type = st.sidebar.selectbox(
        'Select Model',
        (
            'SVC',
            'LogisticRegression',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'RidgeClassifier',
            'SGDClassifier',
            'KNeighborsClassifier',
            'MLPClassifier')
    )
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        if model_type == "SVC":
            model = svc_param_selector()
        elif model_type == 'LogisticRegression':
            model = lr_param_selector()
        elif model_type == 'DecisionTreeClassifier':
            model = dt_param_selector()
        elif model_type == 'RandomForestClassifier':
            model = rf_param_selector()
        elif model_type == 'RidgeClassifier':
            model = rc_param_selector()
        elif model_type == 'SGDClassifier':
            model = sgd_param_selector()
        elif model_type == 'KNeighborsClassifier':
            model = kn_param_selector()
        elif model_type == 'MLPClassifier':
            model = mlp_param_selector()
    return model, model_type

def generate_data_classification(X, Y):
    # split into train and test sets
    size = st.sidebar.slider("Percentage of dataset division",
                     min_value=0.1,
                     max_value=0.9,
                     step = 0.1,
                     value=0.3,
                     help="This is the value which will be used to divide the data for training and testing. Default = 30%")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, X_test, y_train, y_test

def main(model, model_type, X_train, X_test, y_train, y_test):
    if 'classification_accuracy' not in st.session_state:
        st.session_state.classification_accuracy=pd.DataFrame(columns = ['Algorithm', 'Accuracy_(%)'])
    accuracy = st.session_state.classification_accuracy
    model.fit(X_train, y_train.values.ravel())
    y_true, y_pred = y_test, model.predict(X_test)
    train_score = int(model.score(X_train, y_train) * 100)
    test_score = int(100 * model.score(X_test, y_test))
    st.subheader(model_type + ' results')
    st.write('Train Acc:', train_score)
    st.write('Test Acc:', test_score)

    ##Confusion matrix
    target_column = y_test.columns[0]
    fig, ax = plt.subplots()
    data = {'y_Actual': y_test[target_column].values,
            'y_Predicted': y_pred}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrixf = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'],
                                    colnames=['Predicted'])
    sns.heatmap(confusion_matrixf, annot=True)
    st.pyplot(fig)
    st.write(model_type + " Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report: ")
    st.write(classification_report(y_test, y_pred))
    LR_accuracy = metrics.accuracy_score(y_test, y_pred)
    LR_accuracy = LR_accuracy*100
    st.write(model_type + " Classifier Accuracy:", LR_accuracy)
    bool_var = 0
    for index, row in accuracy.iterrows():
        if accuracy.loc[index,'Algorithm'] == str(model_type):
            accuracy.loc[index,'Accuracy_(%)'] = LR_accuracy
            bool_var = 1
    if bool_var == 0:
        accuracy = accuracy.append({'Algorithm':model_type,'Accuracy_(%)':LR_accuracy}, ignore_index=True)
    accuracy = accuracy.sort_values(['Accuracy_(%)'])
    fig2 = px.bar(accuracy, x='Algorithm', y='Accuracy_(%)',title="Accuracy of each Classifier",color='Accuracy_(%)')
    st.session_state.classification_accuracy = accuracy
    st.plotly_chart(fig2)

# @st.cache
def app():
    if 'data' not in st.session_state:
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = st.session_state.data
        st.markdown('# Classification Models')
        options = st.sidebar.selectbox(
            'Options',
            (
                'Select Features and Target',
                'Models',
            )
        )
        if options == 'Select Features and Target':
            st.session_state.X_c, st.session_state.Y_c = Classification_X_Y_Selector(df)
        if options == 'Models':
            try:
                X= st.session_state.X_c
                Y = st.session_state.Y_c
            except:
                st.markdown("Please Select the X and Y through `Select Features and Target` Option!")
                return
            model, model_type = Classification_algos_selector()
            X_train, X_test, y_train, y_test = generate_data_classification(X,Y)
            main(model, model_type, X_train, X_test, y_train, y_test)

