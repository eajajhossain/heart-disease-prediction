import streamlit as st
import numpy as np
import pandas as pd
import pickle


@st.cache_resource
def load_artifacts():
    with open("logistic_heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, scaler, columns


model, scaler, columns = load_artifacts()


st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Predict heart disease risk using clinical data")


age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1=Normal,2=Fixed,3=Reversible)", [1, 2, 3])


if st.button("Predict"):
    # Step 1: Raw input dict
    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # Step 2: DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[columns]

    # Step 3: Scale
    input_scaled = scaler.transform(input_df)

   
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease ({probability:.2%})")
    else:
        st.success(f"✅ Low risk of Heart Disease ({probability:.2%})")
