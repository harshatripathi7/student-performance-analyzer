import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# LOAD DATA
# ==================================================

file_path = "data/raw/student-mat.csv"

data = pd.read_csv(
    file_path,
    sep=";"
)

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ==================================================
# SELECT FEATURES
# ==================================================

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
y = data["G3"]


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/student_performance_model.pkl"
)

print("\nRandom Forest model loaded successfully!")


# ==================================================
# MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# MODEL METRICS
# ==================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n===== MODEL EVALUATION =====")

print(
    f"MAE: {mae:.2f}"
)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    f"R² Score: {r2:.2f}"
)


# ==================================================
# CREATE FIGURES DIRECTORY
# ==================================================

import os

os.makedirs(
    "reports/figures",
    exist_ok=True
)


# ==================================================
# 1. ACTUAL VS PREDICTED
# ==================================================

print("\nCreating Actual vs Predicted plot...")

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred
)

plt.plot(
    [0, 20],
    [0, 20],
    linestyle="--"
)

plt.xlabel(
    "Actual Final Grade"
)

plt.ylabel(
    "Predicted Final Grade"
)

plt.title(
    "Actual vs Predicted Final Grades"
)

plt.xlim(
    0,
    20
)

plt.ylim(
    0,
    20
)

plt.tight_layout()

plt.savefig(
    "reports/figures/actual_vs_predicted.png",
    dpi=150
)

plt.close()

print(
    "Saved actual_vs_predicted.png"
)


# ==================================================
# CALCULATE RESIDUALS
# ==================================================

residuals = y_test - y_pred


# ==================================================
# 2. RESIDUAL PLOT
# ==================================================

print("\nCreating residual plot...")

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Final Grade"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Prediction Residuals"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/residual_plot.png",
    dpi=150
)

plt.close()

print(
    "Saved residual_plot.png"
)


# ==================================================
# 3. ERROR DISTRIBUTION
# ==================================================

print("\nCreating error distribution plot...")

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    residuals,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Distribution of Prediction Errors"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/error_distribution.png",
    dpi=150
)

plt.close()

print(
    "Saved error_distribution.png"
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n===== EVALUATION COMPLETE =====")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R²:",
    round(r2, 2)
)

print(
    "\nEvaluation plots saved successfully!"
)
