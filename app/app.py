import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/student_performance_model.pkl"

model = joblib.load(MODEL_PATH)


# ==================================================
# TITLE
# ==================================================

st.title("🎓 Student Performance Analyzer")

st.write(
    """
    An interactive machine learning application that predicts
    a student's final academic grade based on academic,
    demographic, and lifestyle-related factors.
    """
)


st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📋 Student Information")


age = st.sidebar.slider(
    "Age",
    min_value=15,
    max_value=22,
    value=17
)


studytime = st.sidebar.slider(
    "Weekly Study Time",
    min_value=1,
    max_value=4,
    value=2
)


failures = st.sidebar.slider(
    "Previous Failures",
    min_value=0,
    max_value=3,
    value=0
)


absences = st.sidebar.slider(
    "Absences",
    min_value=0,
    max_value=75,
    value=4
)


G1 = st.sidebar.slider(
    "First Period Grade (G1)",
    min_value=0,
    max_value=20,
    value=12
)


G2 = st.sidebar.slider(
    "Second Period Grade (G2)",
    min_value=0,
    max_value=20,
    value=13
)


Medu = st.sidebar.slider(
    "Mother's Education",
    min_value=0,
    max_value=4,
    value=3
)


Fedu = st.sidebar.slider(
    "Father's Education",
    min_value=0,
    max_value=4,
    value=2
)


freetime = st.sidebar.slider(
    "Free Time",
    min_value=1,
    max_value=5,
    value=3
)


goout = st.sidebar.slider(
    "Going Out",
    min_value=1,
    max_value=5,
    value=3
)


health = st.sidebar.slider(
    "Health",
    min_value=1,
    max_value=5,
    value=4
)


# ==================================================
# INPUT DATA
# ==================================================

input_data = pd.DataFrame(
    {
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "freetime": [freetime],
        "goout": [goout],
        "health": [health]
    }
)


# ==================================================
# PREDICTION
# ==================================================

if st.button(
    "🔮 Predict Final Grade",
    use_container_width=True
):

    prediction = model.predict(
        input_data
    )[0]

    prediction = np.clip(
        prediction,
        0,
        20
    )

    st.session_state["prediction"] = prediction


# ==================================================
# DISPLAY PREDICTION
# ==================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Final Grade",
            f"{prediction:.2f} / 20"
        )

    with col2:

        percentage = (prediction / 20) * 100

        st.metric(
            "Estimated Percentage",
            f"{percentage:.1f}%"
        )

    with col3:

        if prediction >= 16:

            performance = "Excellent"

        elif prediction >= 12:

            performance = "Good"

        elif prediction >= 10:

            performance = "Average"

        else:

            performance = "Needs Improvement"

        st.metric(
            "Performance Level",
            performance
        )


# ==================================================
# RECOMMENDATIONS
# ==================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.subheader(
        "💡 Recommendations"
    )

    recommendations = []

    if studytime <= 2:

        recommendations.append(
            "📚 Consider increasing weekly study time."
        )

    if failures > 0:

        recommendations.append(
            "🎯 Focus on subjects where previous failures occurred."
        )

    if absences > 10:

        recommendations.append(
            "🏫 Reducing absences may help improve academic performance."
        )

    if G1 < 10:

        recommendations.append(
            "📝 Strengthen preparation for upcoming assessments."
        )

    if G2 < 10:

        recommendations.append(
            "📈 Focus on improving recent academic performance."
        )

    if goout >= 4:

        recommendations.append(
            "⚖️ Maintain a healthy balance between social activities and study."
        )

    if not recommendations:

        recommendations.append(
            "🌟 Your current academic indicators look positive. Keep it up!"
        )

    for recommendation in recommendations:

        st.write(
            recommendation
        )


# ==================================================
# MODEL EVALUATION
# ==================================================

st.divider()

st.header(
    "📊 Model Evaluation"
)

st.write(
    """
    The Random Forest model was evaluated on a held-out test set
    containing 79 students.
    """
)


# Metrics

metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.metric(
        "MAE",
        "1.08"
    )


with metric2:

    st.metric(
        "RMSE",
        "1.78"
    )


with metric3:

    st.metric(
        "R² Score",
        "0.85"
    )


st.write(
    """
    **Interpretation:** The model explains approximately 85% of
    the variation in final grades. On average, predictions differ
    from actual grades by about 1.08 points.
    """
)


# ==================================================
# ACTUAL VS PREDICTED
# ==================================================

st.subheader(
    "🎯 Actual vs Predicted Grades"
)

actual_predicted_path = (
    "reports/figures/actual_vs_predicted.png"
)

if os.path.exists(actual_predicted_path):

    st.image(
        actual_predicted_path,
        use_container_width=True
    )

else:

    st.warning(
        "Actual vs Predicted plot not found."
    )


# ==================================================
# RESIDUAL ANALYSIS
# ==================================================

st.subheader(
    "📉 Prediction Residuals"
)

residual_path = (
    "reports/figures/residual_plot.png"
)

if os.path.exists(residual_path):

    st.image(
        residual_path,
        use_container_width=True
    )

else:

    st.warning(
        "Residual plot not found."
    )


# ==================================================
# ERROR DISTRIBUTION
# ==================================================

st.subheader(
    "📊 Prediction Error Distribution"
)

error_path = (
    "reports/figures/error_distribution.png"
)

if os.path.exists(error_path):

    st.image(
        error_path,
        use_container_width=True
    )

else:

    st.warning(
        "Error distribution plot not found."
    )


# ==================================================
# EDA VISUALIZATIONS
# ==================================================

st.divider()

st.header(
    "🔍 Exploratory Data Analysis"
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Final Grade Distribution"
    )

    path = (
        "reports/figures/final_grade_distribution.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )


with col2:

    st.subheader(
        "Study Time vs Final Grade"
    )

    path = (
        "reports/figures/studytime_vs_grade.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )


st.subheader(
    "Previous Failures vs Final Grade"
)

path = (
    "reports/figures/failures_vs_grade.png"
)

if os.path.exists(path):

    st.image(
        path,
        use_container_width=True
    )


# ==================================================
# FEATURE INFORMATION
# ==================================================

st.divider()

st.header(
    "📌 Features Used by the Model"
)

feature_data = pd.DataFrame(
    {
        "Feature": [
            "age",
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "Medu",
            "Fedu",
            "freetime",
            "goout",
            "health"
        ],
        "Description": [
            "Student age",
            "Weekly study time",
            "Number of previous failures",
            "Number of school absences",
            "First period grade",
            "Second period grade",
            "Mother's education level",
            "Father's education level",
            "Free time after school",
            "Frequency of going out",
            "Current health status"
        ]
    }
)

st.dataframe(
    feature_data,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Student Performance Analyzer | "
    "Machine Learning Project | "
    "Random Forest Regression"
)
