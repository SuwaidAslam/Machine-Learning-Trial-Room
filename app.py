import streamlit as st
# from PIL import  Image

# Custom imports 
from multipage import MultiPage
from Pages import data_upload, explore_data, classification_models, regression_models # import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
# display = Image.open('Logo.png')
# display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
# col1, col2 = st.beta_columns(2)
# col1.image(display, width = 400)
# col2.title("Machine Learning Trial Room App")
st.title("Machine Learning Trial Room App")

# Add all your application here
app.add_page("Upload Data", data_upload.app)
app.add_page("Explore Data", explore_data.app)
app.add_page("Classification Models", classification_models.app)
app.add_page("Regression Models", regression_models.app)


# The main app
app.run()

# Footer of the app
footer = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'ML Trial Room - Developed By Suwaid'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(footer, unsafe_allow_html=True)
