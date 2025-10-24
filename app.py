import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title = "ML Model Prediction App 🈸",
    page_icon = "🔮",
    layout = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "This is a header. This is an *extremely* cool app!"
    }
)

# Sidebar for input fields
st.sidebar.title("ML Model Prediction App")
st.sidebar.write("Enter the feature values in the dropdowns on the main page.")

# Contact Info
st.sidebar.subheader("Contact Info ℹ️")
st.sidebar.write("Phone ☎️: +234 706 7159 089")
st.sidebar.write("Email 📧: danielayomideh@gmail.com")

# Tips
st.sidebar.subheader("Tips ➡️")
st.sidebar.write("1. Ensure your data is clean and well-prepared.")
st.sidebar.write("2. Normalize feature values for better model performance.")
st.sidebar.write("3. Validate model performance with a test set.")

# Company Details
st.sidebar.subheader("About Our Company 🎯")
st.sidebar.write("We are a leading provider of innovative AI solutions. Our mission is to harness the power of AI to solve real-world problems.")

# About
st.sidebar.subheader("About This App 🆎")
st.sidebar.write("This app allows users to input feature values and get predictions from a trained machine learning model.")

# Reasons to Own a Bank Account
st.sidebar.subheader("Why Own a Bank Account? 🏦")
st.sidebar.write("1. Secure place to store your money.")
st.sidebar.write("2. Easy access to funds.")
st.sidebar.write("3. Opportunity to earn interest.")
st.sidebar.write("4. Access to financial services such as loans and credit.")

# Add a colorful header and instructions
st.markdown('<h1 style="color: darkblue;">ML Model Prediction App</h1>', unsafe_allow_html=True)
st.write("This is a simple ML model prediction app that predicts the target variable based on the input features.")
st.write("The model is trained on the Financial Inclusion in Africa dataset.")
st.write('The target variable is "bank_account" which is a binary variable indicating whether the respondent has a bank account or not.')
st.write("Enter the feature values in the dropdowns and click the predict button to get the prediction.")

# Load your trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except (EOFError, FileNotFoundError) as e:
    model = None
    st.error(f"Error loading model: {e}")

# Ensure the loaded model has a predict method
if model is None or not hasattr(model, "predict"):
    st.error("Loaded object is not a valid ML model with a predict method.")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    feature_1 = st.selectbox("Country", range(1, 101))
    feature_2 = st.selectbox("Location Type", range(1, 101))
    feature_3 = st.selectbox("Cellphone Access", range(1, 101))
    feature_4 = st.selectbox("Household Size", range(1, 101))
    
with col2:
    feature_5 = st.selectbox("Age of Respondent", range(1, 101))
    feature_6 = st.selectbox("Gender of Respondent", range(1, 101))
  
with col3:
    feature_7 = st.selectbox("Relationship with Head", range(1, 101))
    feature_8 = st.selectbox("Marital Status", range(1, 101))
    feature_9 = st.selectbox("Education Level", range(1, 101))
    feature_10 = st.selectbox("Job Type", range(1, 101))

# Validation button on the main page
if st.button("Predict"):
    features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]])
    if model is not None and hasattr(model, "predict"):
        prediction = model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.error("The loaded object is not a valid ML model with a predict method.")

# Add a footer with additional information        
st.markdown("---")
st.markdown("### Additional Information")
st.write("This app is powered by a machine learning model trained on the Financial Inclusion in Africa dataset.")
st.write("For more information about the dataset, visit the [dataset page](https://www.kaggle.com/competitions/financial-inclusion-in-africa).")
st.write("For any inquiries or support, please contact us at the provided contact information in the sidebar.")

# Footer with additional links and resources
st.markdown("---")
st.markdown("### Useful Links")
st.write("[Streamlit Documentation](https://docs.streamlit.io/)")
st.write("[Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)")
st.write("[NumPy Documentation](https://numpy.org/doc/)")
st.write("[Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)")

# Footer with copyright information
st.markdown("---")
st.markdown("© 2023 Da' Ayomideh's Lib. All rights reserved.")