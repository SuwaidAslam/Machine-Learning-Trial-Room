import streamlit as st
import pandas as pd
import io


# @st.cache
def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv or xlsx file.")
    st.write("\n")

    # Uploading Data
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            st.session_state.data = pd.read_excel(uploaded_file)

    if st.button("Load Data"):
        df = st.session_state.data
        st.write(df)
        buffer = io.StringIO()
        df.info(verbose=True, buf=buffer)
        s = buffer.getvalue()
        st.markdown('### Columns information')
        st.text(s)
