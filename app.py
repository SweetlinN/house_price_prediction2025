import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load Model and Features
# -----------------------------
model = joblib.load("house_price_predictor.pkl")
features = joblib.load("features_joblib.pkl")   # This should be a list of feature names

st.title("🏠 House Price Prediction App")
st.write("Enter the values below to predict the house price.")

# -----------------------------
# Generate Input Fields Dynamically
# -----------------------------
inputs = []

st.subheader("Input Features")
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Convert to numpy array for prediction
input_array = np.array(inputs).reshape(1, -1)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_array)[0]
    st.success(f"💰 Predicted House Price: **rs{prediction:,.2f}**")