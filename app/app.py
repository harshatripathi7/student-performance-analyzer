import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt


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
# FEATURES
# ============================================================

FEATURES = [
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
]


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
# LOAD SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)


explainer = load_shap_explainer()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 Student Performance Analyzer")

st.markdown(
    """
    An interactive machine learning application that predicts a
    student's final academic grade and explains the factors influencing
    the prediction.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Model Information")

st.sidebar.markdown(
    """
    **Model:** Random Forest Regression

    **R² Score:** 0.85

    **MAE:** 1.08

    **RMSE:** 1.78

    **Dataset:** 395 students
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Predictions are estimates generated from patterns learned from "
    "the student performance dataset."
)


# ============================================================
# INPUT SECTION
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
        help=(
            "1 = less than 2 hours, "
            "2 = 2–5 hours, "
            "3 = 5–10 hours, "
            "4 = more than 10 hours"
        )
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
        value=2
    )

    fedu = st.slider(
        "Father's Education",
        min_value=0,
        max_value=4,
        value=2
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

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

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
    # PERSONALIZED RECOMMENDATIONS
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


    # ========================================================
    # SHAP INDIVIDUAL EXPLANATION
    # ========================================================

    st.divider()

    st.header("🧠 Why Did the Model Make This Prediction?")

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) shows how each input
        feature influenced this individual prediction.

        **Positive SHAP values** push the prediction higher.

        **Negative SHAP values** push the prediction lower.
        """
    )

    # Calculate SHAP values for the entered student

    shap_values = explainer.shap_values(input_data)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        individual_shap = shap_values[0][0]

    else:

        individual_shap = shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "Value": input_data.iloc[0].values,
        "SHAP Value": individual_shap
    })

    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    # --------------------------------------------------------
    # TOP POSITIVE / NEGATIVE FACTORS
    # --------------------------------------------------------

    positive_factors = shap_df[
        shap_df["SHAP Value"] > 0
    ].head(3)

    negative_factors = shap_df[
        shap_df["SHAP Value"] < 0
    ].sort_values(
        "SHAP Value"
    ).head(3)

    explanation_col1, explanation_col2 = st.columns(2)

    with explanation_col1:

        st.subheader("🟢 Factors Increasing Prediction")

        if len(positive_factors) == 0:

            st.write("No major positive contributors.")

        else:

            for _, row in positive_factors.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"({row['Value']}) → "
                    f"+{row['SHAP Value']:.3f}"
                )


    with explanation_col2:

        st.subheader("🔴 Factors Decreasing Prediction")

        if len(negative_factors) == 0:

            st.write("No major negative contributors.")

        else:

            for _, row in negative_factors.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"({row['Value']}) → "
                    f"{row['SHAP Value']:.3f}"
                )


    # --------------------------------------------------------
    # SHAP WATERFALL
    # --------------------------------------------------------

    st.subheader("📊 Individual Prediction Explanation")

    try:

        sample_explanation = shap.Explanation(
            values=individual_shap,
            base_values=explainer.expected_value,
            data=input_data.iloc[0].values,
            feature_names=FEATURES
        )

        fig = plt.figure()

        shap.plots.waterfall(
            sample_explanation,
            show=False
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    except Exception as error:

        st.warning(
            f"Could not generate the interactive SHAP waterfall: {error}"
        )


    # --------------------------------------------------------
    # SHAP TABLE
    # --------------------------------------------------------

    with st.expander("🔎 View Detailed SHAP Values"):

        display_df = shap_df[
            ["Feature", "Value", "SHAP Value"]
        ].copy()

        display_df["SHAP Value"] = display_df[
            "SHAP Value"
        ].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


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
    ### Model Comparison

    | Model | MAE | RMSE | R² |
    |---|---:|---:|---:|
    | Linear Regression | 1.38 | 2.16 | 0.77 |
    | **Random Forest** | **1.08** | **1.78** | **0.85** |
    | Gradient Boosting | 1.16 | 1.85 | 0.83 |
    """
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("🔍 Global Feature Importance")

st.markdown(
    """
    This visualization shows which features were most useful to the
    Random Forest model across the overall dataset.

    Feature importance indicates predictive usefulness, not causation.
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


# ============================================================
# GLOBAL SHAP EXPLANATIONS
# ============================================================

st.divider()

st.header("🧠 Global SHAP Explainability")

st.markdown(
    """
    SHAP provides a model-independent way to understand how features
    influence predictions.

    Unlike traditional feature importance, SHAP values show both the
    magnitude and direction of feature contributions.
    """
)


shap_summary_path = os.path.join(
    FIGURES_PATH,
    "shap_summary.png"
)

if os.path.exists(shap_summary_path):

    st.subheader("SHAP Summary")

    st.image(
        shap_summary_path,
        caption="SHAP Summary Plot",
        use_container_width=True
    )


shap_bar_path = os.path.join(
    FIGURES_PATH,
    "shap_feature_importance.png"
)

if os.path.exists(shap_bar_path):

    st.subheader("Mean Absolute SHAP Importance")

    st.image(
        shap_bar_path,
        caption="Mean Absolute SHAP Feature Importance",
        use_container_width=True
    )


st.info(
    "G2 is strongly predictive because it represents the student's "
    "second-period grade, which occurs shortly before the final grade G3."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header("📈 Model Evaluation")

st.markdown(
    """
    These visualizations evaluate the Random Forest model on unseen
    test data.
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
    during exploratory analysis.
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
    "Machine Learning & Explainable AI Project by Harsha Tripathi"
)
