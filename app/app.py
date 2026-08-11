import streamlit as st
import joblib
import pandas as pd


# Page configuration
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="centered"
)


# Load trained model
model = joblib.load("models/student_performance_model.pkl")


# Title
st.title("🎓 Student Performance Analyzer")

st.write(
    "Predict a student's final academic grade using "
    "a machine learning model."
)

st.divider()


# Student information
st.header("Student Information")

age = st.number_input(
    "Age",
    min_value=15,
    max_value=22,
    value=17
)

studytime = st.selectbox(
    "Study Time",
    options=[1, 2, 3, 4],
    index=1,
    help="1 = Very low, 2 = Low, 3 = High, 4 = Very high"
)

failures = st.selectbox(
    "Previous Failures",
    options=[0, 1, 2, 3],
    index=0
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=75,
    value=4
)


# Academic performance
st.header("Academic Performance")

G1 = st.number_input(
    "First Period Grade (G1)",
    min_value=0,
    max_value=20,
    value=12
)

G2 = st.number_input(
    "Second Period Grade (G2)",
    min_value=0,
    max_value=20,
    value=13
)


# Family and lifestyle information
st.header("Family & Lifestyle")

Medu = st.selectbox(
    "Mother's Education Level",
    options=[0, 1, 2, 3, 4],
    index=3
)

Fedu = st.selectbox(
    "Father's Education Level",
    options=[0, 1, 2, 3, 4],
    index=2
)

freetime = st.slider(
    "Free Time",
    min_value=1,
    max_value=5,
    value=3
)

goout = st.slider(
    "Going Out",
    min_value=1,
    max_value=5,
    value=3
)

health = st.slider(
    "Health",
    min_value=1,
    max_value=5,
    value=4
)


st.divider()


# Prediction button
if st.button("🔮 Predict Final Grade"):

    student = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": G1,
        "G2": G2,
        "Medu": Medu,
        "Fedu": Fedu,
        "freetime": freetime,
        "goout": goout,
        "health": health
    }

    student_data = pd.DataFrame([student])

    prediction = model.predict(student_data)[0]

    prediction = max(0, min(20, prediction))

    st.success(
        f"🎯 Predicted Final Grade: {prediction:.2f} / 20"
    )