import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# App Title
st.title("ML Model Prediction App")

# User Inputs (example: 3 features)
st.subheader("Enter input features:")
feature1 = st.number_input("Feature 1", min_value=0.0)
feature2 = st.number_input("Feature 2", min_value=0.0)
feature3 = st.number_input("Feature 3", min_value=0.0)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")