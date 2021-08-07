import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# @st.cache
def app():
    if 'data' not in st.session_state:
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = st.session_state.data
        st.markdown('# Explore Data')
        st.write(df)
        col = st.selectbox('Select Column', df.columns.tolist())
        fig_0 = plt.figure(figsize=(10, 7))
        plt.boxplot(df[col])
        st.pyplot(fig_0)

        fig, ax = plt.subplots()
        corr = df.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
