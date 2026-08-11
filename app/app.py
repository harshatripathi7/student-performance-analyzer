import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/student_performance_model.pkl"
    )


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/raw/student-mat.csv",
        sep=";"
    )


model = load_model()
data = load_data()


# ==================================================
# HEADER
# ==================================================

st.title("🎓 Student Performance Analyzer")

st.markdown(
    """
    ### Analyze student performance and predict final grades

    This machine learning application analyzes academic,
    demographic, and lifestyle-related factors to predict
    a student's final grade out of 20.
    """
)

st.divider()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🎯 Grade Prediction",
        "📊 Data Analysis"
    ]
)


# ==================================================
# PAGE 1 — GRADE PREDICTION
# ==================================================

if page == "🎯 Grade Prediction":

    st.header("👤 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=25,
            value=17
        )

    with col2:
        studytime = st.slider(
            "Weekly Study Time",
            min_value=1,
            max_value=4,
            value=2
        )

    with col3:
        failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=3,
            value=0
        )


    # ==================================================
    # ACADEMIC INFORMATION
    # ==================================================

    st.header("📚 Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        absences = st.number_input(
            "School Absences",
            min_value=0,
            max_value=100,
            value=4
        )

    with col2:
        G1 = st.number_input(
            "First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=12
        )

    with col3:
        G2 = st.number_input(
            "Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=13
        )


    # ==================================================
    # FAMILY AND LIFESTYLE
    # ==================================================

    st.header("🏠 Family & Lifestyle")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Medu = st.slider(
            "Mother's Education",
            min_value=0,
            max_value=4,
            value=2
        )

    with col2:
        Fedu = st.slider(
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

    with col4:
        goout = st.slider(
            "Going Out",
            min_value=1,
            max_value=5,
            value=3
        )

    health = st.slider(
        "Health Status",
        min_value=1,
        max_value=5,
        value=3
    )


    st.divider()


    # ==================================================
    # PREDICTION
    # ==================================================

    if st.button(
        "🎯 Predict Final Grade",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                studytime,
                failures,
                absences,
                G1,
                G2,
                Medu,
                Fedu,
                freetime,
                goout,
                health
            ]],
            columns=[
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
        )

        prediction = model.predict(
            input_data
        )[0]

        # Keep prediction within valid grade range
        prediction = max(
            0,
            min(20, prediction)
        )

        st.success(
            "Prediction generated successfully!"
        )

        st.metric(
            "Predicted Final Grade",
            f"{prediction:.2f} / 20"
        )


        # ==================================================
        # PERFORMANCE INTERPRETATION
        # ==================================================

        if prediction >= 16:

            st.success(
                "🌟 Excellent predicted performance!"
            )

        elif prediction >= 12:

            st.info(
                "👍 Good predicted performance."
            )

        elif prediction >= 10:

            st.warning(
                "📚 Moderate predicted performance. "
                "There may be room for improvement."
            )

        else:

            st.error(
                "⚠️ Low predicted performance. "
                "Additional academic support may be beneficial."
            )


    # ==================================================
    # MODEL INFORMATION
    # ==================================================

    st.divider()

    st.header("🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:

        st.metric(
            "R² Score",
            "0.85"
        )

    with col3:

        st.metric(
            "MAE",
            "1.08"
        )

    st.caption(
        "Random Forest model trained on 395 student records "
        "using 11 selected features."
    )


# ==================================================
# PAGE 2 — DATA ANALYSIS
# ==================================================

else:

    st.header("📊 Exploratory Data Analysis")

    st.markdown(
        """
        Explore the student dataset and examine relationships
        between academic performance and student characteristics.
        """
    )


    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    st.subheader(
        "📋 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Students",
            data.shape[0]
        )

    with col2:

        st.metric(
            "Features",
            data.shape[1]
        )

    with col3:

        st.metric(
            "Average Grade",
            f"{data['G3'].mean():.2f}"
        )

    with col4:

        st.metric(
            "Median Grade",
            f"{data['G3'].median():.0f}"
        )


    st.divider()


    # ==================================================
    # FINAL GRADE DISTRIBUTION
    # ==================================================

    st.subheader(
        "📈 Final Grade Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        data["G3"],
        bins=11,
        edgecolor="black"
    )

    ax.set_title(
        "Distribution of Final Grades"
    )

    ax.set_xlabel(
        "Final Grade (G3)"
    )

    ax.set_ylabel(
        "Number of Students"
    )

    st.pyplot(fig)


    # ==================================================
    # STUDY TIME VS FINAL GRADE
    # ==================================================

    st.subheader(
        "📚 Study Time vs Final Grade"
    )

    studytime_avg = (
        data.groupby("studytime")["G3"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        studytime_avg.index.astype(str),
        studytime_avg.values
    )

    ax.set_xlabel(
        "Study Time Level"
    )

    ax.set_ylabel(
        "Average Final Grade"
    )

    ax.set_title(
        "Average Final Grade by Study Time"
    )

    st.pyplot(fig)


    # ==================================================
    # FAILURES VS FINAL GRADE
    # ==================================================

    st.subheader(
        "⚠️ Previous Failures vs Final Grade"
    )

    failures_avg = (
        data.groupby("failures")["G3"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        failures_avg.index.astype(str),
        failures_avg.values
    )

    ax.set_xlabel(
        "Previous Failures"
    )

    ax.set_ylabel(
        "Average Final Grade"
    )

    ax.set_title(
        "Average Final Grade by Previous Failures"
    )

    st.pyplot(fig)


    # ==================================================
    # DATASET PREVIEW
    # ==================================================

    st.divider()

    st.subheader(
        "🔎 Dataset Preview"
    )

    st.dataframe(
        data.head(20),
        use_container_width=True
    )


    # ==================================================
    # STATISTICAL SUMMARY
    # ==================================================

    st.subheader(
        "📊 Statistical Summary"
    )

    st.dataframe(
        data.describe(),
        use_container_width=True
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "🎓 Student Performance Analyzer | "
    "Machine Learning Project"
)
