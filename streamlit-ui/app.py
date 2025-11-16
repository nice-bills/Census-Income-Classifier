import streamlit as st
import requests

API_URL = "http://localhost:9696/predict"   # update if deployed

st.title("Income Prediction App")

st.write("Enter the details below:")

# Example fields (modify based on your preprocessing & model inputs)
age = st.number_input("Age", 0, 120, 30)
education = st.selectbox("Education", ["Bachelors", "HS-grad", "Masters", "Some-college"])
hours_per_week = st.number_input("Hours per week", 1, 80, 40)

if st.button("Predict"):
    payload = {
        "age": age,
        "education": education,
        "hours_per_week": hours_per_week,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Income: {prediction}")
    else:
        st.error("Error making prediction")
