[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/suwaidaslam/machine-learning-trial-room/main/app.py)
# Machine Learning Trial Room
An application to try different machine learning models on preprocessed data directly from the browser.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![pythonbadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

# Demo


https://user-images.githubusercontent.com/45914161/132569404-64c99027-fa68-4b2a-9a51-82c7ac0d73bf.mp4



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
$ source env/Source/activate
```
- Install dependencies using
```
$ pip install -r requirements.txt
```
- Run Streamlit
```
$ streamlit run app.py
```
## Contributions are welcome!

Feel free to open a pull request or an issue if you're thinking of a feature you'd like to see in the app.

Off the top of my head, I can think of:

- [ ] Adding Preprocessing Part in the app, so we do not have to preprocess our dataset outside the app.
- [ ] Adding  feature engineering part

But if you've got other ideas, I will be happy to discuss them with you.

## Contact

For any feedback or queries, please reach out to me at [suwaidaslam@gmail.com](suwaidaslam@gmail.com).
