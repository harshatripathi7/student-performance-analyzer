import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
file_path = "data/raw/student-mat.csv"

data = pd.read_csv(file_path, sep=";")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# Features used by the model
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

# Load trained Random Forest model
model = joblib.load("models/student_performance_model.pkl")

print("\nRandom Forest model loaded successfully!")

# Get feature importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance_df.sort_values(
    by="Importance",
    ascending=False
))

# Create visualization
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

# Save figure
output_path = "reports/figures/feature_importance.png"

plt.savefig(output_path, dpi=300)

plt.close()

print("\nFeature importance plot saved successfully!")
print(f"Saved to: {output_path}")
