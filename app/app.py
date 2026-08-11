import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/student_performance_model.pkl"
DATA_PATH = "data/raw/student-mat.csv"
FIGURES_PATH = "reports/figures"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


data = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Performance Analyzer")

st.markdown(
    """
    This machine learning application predicts a student's final grade
    out of **20** using academic, demographic, and lifestyle-related factors.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📚 About the Model")

st.sidebar.markdown(
    """
    **Algorithm:** Random Forest Regression

    **R² Score:** 0.85

    **MAE:** 1.08

    **RMSE:** 1.78

    The model was trained using 395 student records.
    """
)

st.sidebar.divider()

st.sidebar.info(
    "The prediction is an estimate based on patterns learned from the "
    "student performance dataset."
)


# ============================================================
# STUDENT INPUT
# ============================================================

st.header("📝 Enter Student Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17,
        step=1
    )

    studytime = st.slider(
        "Weekly Study Time",
        min_value=1,
        max_value=4,
        value=2,
        help="1 = less than 2 hours, 2 = 2–5 hours, "
             "3 = 5–10 hours, 4 = more than 10 hours"
    )

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )

    absences = st.number_input(
        "School Absences",
        min_value=0,
        max_value=93,
        value=4,
        step=1
    )


with col2:

    g1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=12,
        step=1
    )

    g2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=13,
        step=1
    )

    medu = st.slider(
        "Mother's Education",
        min_value=0,
        max_value=4,
        value=2,
        help="0 = none, 1 = primary, 2 = 5th–9th grade, "
             "3 = secondary, 4 = higher education"
    )

    fedu = st.slider(
        "Father's Education",
        min_value=0,
        max_value=4,
        value=2,
        help="0 = none, 1 = primary, 2 = 5th–9th grade, "
             "3 = secondary, 4 = higher education"
    )


with col3:

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


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Final Grade",
    type="primary",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [g1],
        "G2": [g2],
        "Medu": [medu],
        "Fedu": [fedu],
        "freetime": [freetime],
        "goout": [goout],
        "health": [health]
    })

    prediction = model.predict(input_data)[0]

    # Keep prediction within the valid grade range
    prediction = max(0, min(20, prediction))

    percentage = (prediction / 20) * 100

    st.success("Prediction generated successfully!")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Final Grade",
            f"{prediction:.2f} / 20"
        )

    with result_col2:

        st.metric(
            "Estimated Percentage",
            f"{percentage:.1f}%"
        )

    # --------------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # --------------------------------------------------------

    st.subheader("📊 Performance Interpretation")

    if prediction >= 16:

        st.success(
            "🌟 Excellent predicted performance! "
            "The student is expected to perform very strongly."
        )

    elif prediction >= 12:

        st.info(
            "👍 Good predicted performance. "
            "Maintaining consistent study habits could improve the result further."
        )

    elif prediction >= 10:

        st.warning(
            "📚 Moderate predicted performance. "
            "Additional study time and improved academic consistency may help."
        )

    else:

        st.error(
            "⚠️ The predicted grade is relatively low. "
            "The student may benefit from additional academic support."
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("💡 Personalized Recommendations")

    recommendations = []

    if studytime <= 2:
        recommendations.append(
            "📖 Increase weekly study time and maintain a consistent study schedule."
        )

    if failures > 0:
        recommendations.append(
            "🎯 Focus on subjects where previous difficulties occurred."
        )

    if absences > 10:
        recommendations.append(
            "🏫 Reduce unnecessary absences and attend classes consistently."
        )

    if g1 < 10:
        recommendations.append(
            "📝 Strengthen fundamentals to improve performance in upcoming assessments."
        )

    if g2 < 10:
        recommendations.append(
            "📈 Focus strongly on improving current academic performance."
        )

    if goout >= 4:
        recommendations.append(
            "⏰ Balance social activities with dedicated study time."
        )

    if health <= 2:
        recommendations.append(
            "❤️ Pay attention to health and maintain healthy daily routines."
        )

    if not recommendations:
        recommendations.append(
            "🌟 Keep maintaining the current academic habits and consistency."
        )

    for recommendation in recommendations:
        st.write(recommendation)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("🤖 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "R² Score",
        "0.85"
    )

with metric2:
    st.metric(
        "MAE",
        "1.08"
    )

with metric3:
    st.metric(
        "RMSE",
        "1.78"
    )

st.markdown(
    """
    The Random Forest model was selected after comparing three regression
    algorithms.

    | Model | MAE | RMSE | R² |
    |---|---:|---:|---:|
    | Linear Regression | 1.38 | 2.16 | 0.77 |
    | **Random Forest** | **1.08** | **1.78** | **0.85** |
    | Gradient Boosting | 1.16 | 1.85 | 0.83 |
    """
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("🔍 Feature Importance")

st.markdown(
    """
    Feature importance shows which variables were most useful to the
    Random Forest model when predicting the final grade.

    **Important:** Feature importance indicates predictive usefulness,
    not causation.
    """
)

feature_importance_path = os.path.join(
    FIGURES_PATH,
    "feature_importance.png"
)

if os.path.exists(feature_importance_path):

    st.image(
        feature_importance_path,
        caption="Random Forest Feature Importance",
        use_container_width=True
    )

else:

    st.warning(
        "Feature importance visualization is not available."
    )


st.info(
    "G2 has the highest predictive importance in this model. "
    "This is expected because G2 is the student's second-period grade "
    "and is strongly related to the final grade G3."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header("📈 Model Evaluation")

st.markdown(
    """
    The following visualizations evaluate how well the Random Forest
    model performs on unseen test data.
    """
)


# Actual vs Predicted

actual_predicted_path = os.path.join(
    FIGURES_PATH,
    "actual_vs_predicted.png"
)

if os.path.exists(actual_predicted_path):

    st.subheader("🎯 Actual vs Predicted Grades")

    st.image(
        actual_predicted_path,
        caption="Actual vs Predicted Grades",
        use_container_width=True
    )


# Residual Plot

residual_path = os.path.join(
    FIGURES_PATH,
    "residual_plot.png"
)

if os.path.exists(residual_path):

    st.subheader("📉 Residual Plot")

    st.image(
        residual_path,
        caption="Residual Plot",
        use_container_width=True
    )


# Error Distribution

error_path = os.path.join(
    FIGURES_PATH,
    "error_distribution.png"
)

if os.path.exists(error_path):

    st.subheader("📊 Prediction Error Distribution")

    st.image(
        error_path,
        caption="Prediction Error Distribution",
        use_container_width=True
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

st.divider()

st.header("🔬 Exploratory Data Analysis")

st.markdown(
    """
    These visualizations show important relationships discovered
    during exploratory analysis of the student dataset.
    """
)


# Final Grade Distribution

grade_distribution_path = os.path.join(
    FIGURES_PATH,
    "final_grade_distribution.png"
)

if os.path.exists(grade_distribution_path):

    st.subheader("📊 Final Grade Distribution")

    st.image(
        grade_distribution_path,
        caption="Distribution of Final Grades",
        use_container_width=True
    )


# Study Time vs Grade

studytime_path = os.path.join(
    FIGURES_PATH,
    "studytime_vs_grade.png"
)

if os.path.exists(studytime_path):

    st.subheader("📚 Study Time vs Final Grade")

    st.image(
        studytime_path,
        caption="Study Time vs Final Grade",
        use_container_width=True
    )


# Failures vs Grade

failures_path = os.path.join(
    FIGURES_PATH,
    "failures_vs_grade.png"
)

if os.path.exists(failures_path):

    st.subheader("❌ Previous Failures vs Final Grade")

    st.image(
        failures_path,
        caption="Previous Failures vs Final Grade",
        use_container_width=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.header("📚 Dataset Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "Students",
        f"{len(data):,}"
    )

with info_col2:
    st.metric(
        "Features",
        f"{data.shape[1]}"
    )

with info_col3:
    st.metric(
        "Target",
        "G3 / 20"
    )


with st.expander("🔎 View Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Student Performance Analyzer | "
    "Machine Learning Project by Harsha Tripathi"
)
