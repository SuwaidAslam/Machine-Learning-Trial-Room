# Machine-Learning-Trial-Room
An application to try different machine learning models on preprocessed data directly from the browser.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![pythonbadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

# 📱 ML Trial Room 📉
## Introduction 
 ML Trial Room is a **streamlit** application that allows you to play with machine learning models from your browser.

So if you're a data science practitioner you should definitely try it out :wink:

# How does it work ?

1. 🗂️ You upload the preprocessed **dataset**, means dataset must be numerical and does not have NULL values.
2. 📊 Basic Data analysis or Data Exploration.
3. ⚙️ You select X features and Y target label for either Regression Models or Classification Models.
4. 🤖 You select a **model** set its hyper-parameters. You can pick a model from many different models.
5. 📉 The app automatically displays the following results:
   - For Classification Part
      - Train Accuracy
      - Test Accuracy
      - Confusion Matrix
      - Classification Report
      - Model Accuracy
      - Graph of Accuracy of each model Combined
   - For Regression Part
      - R2 Score
      - Mean Squared Error
      - Figure Matching the actual and predicted values
      - Graph of MSE of each model Combined

## Technology Stack 

1. Python 
2. Streamlit 
3. Pandas
4. Scikit-Learn
5. Seaborn

# How to Run 

- Clone the repository
- Setup Virtual environment
```
$ python3 -m venv env
```
- Activate the virtual environment
```
$ source env/bin/activate
```
- Install dependencies using
```
$ pip install -r requirements.txt
```
- Run Streamlit
```
$ streamlit run app.py
```
## Contact

For any feedback or queries, please reach out to [suwaidaslam@gmail.com](suwaidaslam@gmail.com).
