import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/raw/student-mat.csv",
    sep=";"
)

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ============================================================
# FEATURES
# ============================================================

features = [
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

X = data[features]


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/student_performance_model.pkl"
)

print("\nRandom Forest model loaded successfully!")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("\nSHAP values calculated successfully!")


# ============================================================
# GLOBAL SHAP SUMMARY
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.title(
    "SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSHAP summary plot saved successfully!"
)


# ============================================================
# BAR SUMMARY
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.title(
    "Mean Absolute SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP feature importance plot saved successfully!"
)


# ============================================================
# SAMPLE STUDENT EXPLANATION
# ============================================================

sample_index = 0

sample = X.iloc[[sample_index]]

prediction = model.predict(sample)[0]

sample_shap_values = shap_values[sample_index]

explanation = pd.DataFrame({
    "Feature": features,
    "Feature Value": sample.iloc[0].values,
    "SHAP Value": sample_shap_values
})

explanation["Absolute SHAP"] = (
    explanation["SHAP Value"].abs()
)

explanation = explanation.sort_values(
    by="Absolute SHAP",
    ascending=False
)

print("\n===== SAMPLE STUDENT =====")

print(
    sample.to_string(index=False)
)

print(
    f"\nPredicted Final Grade: {prediction:.2f}"
)

print(
    "\n===== SHAP EXPLANATION ====="
)

print(
    explanation[
        ["Feature", "Feature Value", "SHAP Value"]
    ].to_string(index=False)
)


# ============================================================
# WATERFALL PLOT
# ============================================================

sample_explanation = shap.Explanation(
    values=sample_shap_values,
    base_values=explainer.expected_value,
    data=sample.iloc[0].values,
    feature_names=features
)

plt.figure()

shap.plots.waterfall(
    sample_explanation,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSHAP waterfall plot saved successfully!"
)
