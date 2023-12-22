import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import shap


# Load the trained model
model = keras.models.load_model("Admission_Model.h5")  # Replace with the path to your saved model

# Load your training data
# Assuming 'X_train' is your training data
# You need to have this data to fit the scaler
X_train = pd.read_csv("Admission_Predict.csv")  # Replace with the path to your training data

# Define the feature names explicitly
feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

# Fit the scaler on your training data
scaler = MinMaxScaler()
scaler.fit(X_train[feature_names])


# Function to preprocess input data
def preprocess_input_data(input_data):
    # Scale the input data using the fitted scaler
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# Function to check user authentication
def authenticate_user(username, password):
    # Hardcoded username and password (replace with a secure authentication mechanism)
    valid_username = "admin"
    valid_password = "password"

    if username == valid_username and password == valid_password:
        st.session_state.authenticated = True
    else:
        st.warning("Invalid username or password. Please try again.")

# Function to get feature importances
def get_feature_importances(model, feature_names):
    importance = model.feature_importances_  # Replace with the method for getting feature importances
    feature_importance_df = pd.DataFrame(list(zip(feature_names, importance)), columns=['Feature', 'Importance'])
    return feature_importance_df

# Function to get SHAP values for a sample
def get_shap_values(model, sample, feature_names, X_train):
    # Use a KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(sample)
    shap_df = pd.DataFrame(list(zip(feature_names, shap_values[0])), columns=['Feature', 'SHAP Value'])
    return shap_df

# Streamlit app
def main():
    st.title("Admission Prediction Web App")

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Check user authentication
    if not st.session_state.authenticated:
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            authenticate_user(username, password)

        if st.session_state.authenticated:
            st.success("Login successful! You are now authenticated.")
        else:
            return

    # Rest of the code for the app (user input, predictions, etc.)
    st.sidebar.header("User Input")


    # Add dark mode option
    dark_mode = st.checkbox("Dark Mode")

    # Set theme based on dark mode
    if dark_mode:
        st.markdown(
            """
            <style>
                body {
                    color: white;
                    background-color: #222;
                }
                h1, h2, h3 {
                    color: #FFD700;
                }
                .stButton {
                    color: black;
                    background-color: #FFD700;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Data Exploration
    st.subheader("Data Exploration")
    if st.checkbox("Show Raw Data"):
        st.write(X_train)
        # Display feature importance
        st.subheader("Feature Importance Visualization")
        feature_importance_df = get_feature_importances(model, feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature',
                    data=feature_importance_df.sort_values(by='Importance', ascending=False))
        st.pyplot()

    # Collect user input features
    gre_score = st.sidebar.slider("GRE Score", 290, 340, 300)
    toefl_score = st.sidebar.slider("TOEFL Score", 92, 120, 105)
    university_rating = st.sidebar.slider("University Rating", 1, 5, 3)
    sop = st.sidebar.slider("SOP", 1.0, 5.0, 3.0)
    lor = st.sidebar.slider("LOR", 1.0, 5.0, 3.0)
    cgpa = st.sidebar.slider("CGPA", 6.8, 9.92, 8.0)
    research = st.sidebar.selectbox("Research Experience", [0, 1], index=1)

    # Create a DataFrame with the user input
    input_data = pd.DataFrame(
        {
            "GRE Score": [gre_score],
            "TOEFL Score": [toefl_score],
            "University Rating": [university_rating],
            "SOP": [sop],
            "LOR": [lor],  # Note the space after 'LOR'
            "CGPA": [cgpa],
            "Research": [research],
        }
    )

    # Preprocess the input data
    input_data_scaled = preprocess_input_data(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    st.subheader("Prediction")
    st.write(f"The chance of admission is: {prediction[0, 0]:.2%}")

    # Optionally, you can display the input data
    st.subheader("Input Data")
    st.write(input_data)

    # Hyperparameter tuning section
    st.subheader("Model Hyperparameter Tuning")

    # Dropdown for hyperparameter selection
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

    # Display selected hyperparameter value
    st.write("Selected Learning Rate:", learning_rate)

    # Dynamic Feature Selection section
    st.subheader("Dynamic Feature Selection")

    # Allow users to dynamically select features for prediction
    selected_features = st.multiselect("Select Features for Prediction", feature_names, default=feature_names)

    # Create a dynamic input vector based on selected features
    dynamic_input = [st.number_input(f"Enter {feature}", value=0.0) for feature in selected_features]
    dynamic_input_scaled = preprocess_input_data(np.array(dynamic_input).reshape(1, -1))

    # Make prediction and get confidence for dynamic input
    dynamic_prediction = model.predict(dynamic_input_scaled)
    dynamic_confidence = np.max(dynamic_prediction)

    # Display prediction and confidence for dynamic input
    st.write(f"Prediction Confidence (Dynamic): {dynamic_confidence}")

    # Display SHAP values for a sample
    st.subheader("SHAP Values for a Sample")
    sample_index = np.random.randint(0, len(X_train))
    sample = preprocess_input_data(X_train.iloc[sample_index, :].values.reshape(1, -1))

    # Ensure the training data (X_train) is in the correct format
    shap_df = get_shap_values(model, sample, feature_names, X_train)

    st.write("SHAP Values for a Sample:")
    st.write(shap_df)

    # Export results
    st.subheader("Export Results")
    export_button = st.button("Export Results")

if __name__ == "__main__":
    main()
