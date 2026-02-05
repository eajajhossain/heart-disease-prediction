# heart-disease-prediction
Developed an end-to-end machine learning application that predicts the risk of heart disease using patient clinical data. The system is powered by a Logistic Regression model trained with proper preprocessing and deployed as an interactive Streamlit web application for real-time inference
# ❤️ Heart Disease Prediction Web App

An end-to-end machine learning web application that predicts the likelihood of heart disease based on patient clinical parameters.  
The model is trained using **Logistic Regression** and deployed using **Streamlit** for real-time user interaction.

---



---

## 📌 Features
- Predicts heart disease risk using clinical health data
- Interactive and user-friendly Streamlit UI
- Displays both prediction and probability score
- Lightweight and fast inference suitable for real-time use

---

## 🧠 Machine Learning Model
- Algorithm: Logistic Regression
- Task: Binary Classification (Heart Disease: Yes / No)
- Model trained with proper preprocessing to ensure reliable predictions
- Deployed as a reusable inference pipeline

---

## 📊 Input Parameters
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate  
- Exercise Induced Angina  
- ST Depression (Oldpeak)  
- Slope  
- Number of Major Vessels  
- Thalassemia  

> ⚠️ Feature order and preprocessing strictly match the training pipeline to avoid prediction inconsistencies.

---

## 🛠 Tech Stack
- Python
- Scikit-learn
- NumPy
- Pandas
- Streamlit

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/heart-disease-prediction-streamlit-app.git
cd heart-disease-prediction-streamlit-app
pip install -r requirements.txt
streamlit run app.py
