# Load Python libraries
import streamlit as st
import pandas as pd
from joblib import load
# Load models for making predictions
models = {
    'TenYearCHD': load("https://github.com/ro-drick/Heart-Disease-Prediction/blob/master/Heart-Disease-Prediction/models/best_model_TenYearCHD.joblib"),
    'prevalentStroke': load("https://github.com/ro-drick/Heart-Disease-Prediction/tree/master/Heart-Disease-Prediction/models/best_model_prevalentStroke.joblib"),
    'prevalentHyp': load("https://github.com/ro-drick/Heart-Disease-Prediction/tree/master/Heart-Disease-Prediction/models/best_model_prevalentHyp.joblib"),
    'diabetes': load("https://github.com/ro-drick/Heart-Disease-Prediction/tree/master/Heart-Disease-Prediction/models/best_model_diabetes.joblib")
}

feature_names = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 
    'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD' 
]
# Function to get user information
def get_user_input():
    gender_map = {'Male': 1, 'Female': 0}
    education_map = {
        'Some High School': 1,
        'High School Certificate': 2,
        'Some College': 3,
        'College Degree': 4
    }
    smoking_map = {'Yes': 1, 'No': 0}
    bp_meds_map = {'Yes': 1, 'No': 0}

    gender = st.selectbox('Gender', list(gender_map.keys()))
    age = st.slider('Age', 20, 100, 50)
    education = st.selectbox('Education Level', list(education_map.keys()))
    current_smoker = st.selectbox('Current Smoker', list(smoking_map.keys()))
    cigs_per_day = st.slider('Cigarettes Per Day', 0, 100, 10)
    bp_meds = st.selectbox('On Blood Pressure Medication', list(bp_meds_map.keys()))
    tot_chol = st.slider('Total Cholesterol', 100, 400, 200)
    sys_bp = st.slider('Systolic Blood Pressure', 90, 200, 120)
    dia_bp = st.slider('Diastolic Blood Pressure', 60, 130, 80)
    bmi = st.slider('Body Mass Index', 15.0, 50.0, 25.0)
    heart_rate = st.slider('Heart Rate', 40, 120, 75)
    glucose = st.slider('Glucose Level', 50, 200, 80)

    user_data = {
        'male': gender_map[gender],
        'age': age,
        'education': education_map[education],
        'currentSmoker': smoking_map[current_smoker],
        'cigsPerDay': cigs_per_day,
        'BPMeds': bp_meds_map[bp_meds],
        'prevalentStroke': 0,
        'prevalentHyp': 0,
        'diabetes': 0, 
        'totChol': tot_chol,
        'sysBP': sys_bp,
        'diaBP': dia_bp,
        'BMI': bmi,
        'heartRate': heart_rate,
        'glucose': glucose,
        'TenYearCHD': 0 
    }

    return pd.DataFrame(user_data, index=[0])

st.title('Heart Disease Prediction Based on Framingham Heart Study')

st.header('Enter Your Information')
user_input = get_user_input()

user_input = user_input[feature_names]
# Making predictions using the models
if st.button('Make Predictions'):
    predictions = {}
    for outcome, model in models.items():
        predictions[outcome] = model.predict(user_input.drop(columns=[outcome]))[0]
# Displaying predictions
    st.subheader('Your Predictions:')
    for outcome, prediction in predictions.items():
        st.write(f"{outcome}: {'Yes' if prediction == 1 else 'No'}")
