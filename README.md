# Financial Inclusion in Africa - ML Prediction App

# 🎯 What You're Aiming For
This project demonstrates how to predict which individuals are most likely to have or use a bank account using the 'Financial Inclusion in Africa' dataset.

### ➡️ Dataset Link
[Dataset Link](https://drive.google.com/file/d/1FrFTfUln67599LTm2uMTSqM8DjqpAaKL/view?usp=sharing)

### ➡️ Columns Explanation
[Columns Explanation Link](https://drive.google.com/file/d/1jrnrNiJDtff4IIz6lFDwerDhaBKPQwQx/view?usp=sharing).

# ℹ️ Instructions
1. **Install the necessary packages**:
2. **Import your data and perform basic data exploration phase**:
3. **Display general information about the dataset**:
4. **Create a pandas profiling report to gain insights into the dataset**:
5. **Handle Missing and corrupted values**:
6. **Remove duplicates, if they exist**:
7. **Handle outliers, if they exist**:
    - Implement your strategy to handle outliers
8. **Encode categorical features**:
9. **Based on the previous data exploration, train and test a machine learning classifier**:
10. **Create a Streamlit application (locally) and add input fields for your features and a validation button at the end of the form**:
11. **Deploy your application on Streamlit share**:
    - Create a GitHub and a Streamlit Share account.
    - Create a new git repo.
    - Upload your local code to the newly created git repo.
    - Log in to your Streamlit account and deploy your application from the git repo.

### Example Code
```python
pip install pandas scikit-learn streamlit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('path_to_your_dataset.csv')
print(data.head())
print(data.info())
le = LabelEncoder()
data['encoded_column'] = le.fit_transform(data['categorical_column'])
features = data.drop('target_column', axis=1)
label = data['target_column']
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model trained successfully")
```
