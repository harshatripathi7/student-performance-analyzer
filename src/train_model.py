import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

file_path = "data/raw/student-mat.csv"

data = pd.read_csv(file_path, sep=";")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# --------------------------------------------------
# Select features and target
# --------------------------------------------------

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

print("\nFeatures selected:")
print(features)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== DATA SPLIT =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# Define models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# Train and evaluate models
# --------------------------------------------------

results = {}

print("\n===== MODEL COMPARISON =====")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"{name} trained successfully!")

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


# --------------------------------------------------
# Display comparison
# --------------------------------------------------

print("\n===== FINAL MODEL COMPARISON =====")

for name, result in results.items():

    print(
        f"{name:20s} | "
        f"MAE: {result['MAE']:.2f} | "
        f"RMSE: {result['RMSE']:.2f} | "
        f"R²: {result['R2']:.2f}"
    )


# --------------------------------------------------
# Select best model
# --------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["R2"]
)

best_model = results[best_model_name]["model"]

print("\n===== BEST MODEL =====")
print("Best model:", best_model_name)
print(
    f"Best R² Score: "
    f"{results[best_model_name]['R2']:.2f}"
)


# --------------------------------------------------
# Predictions from best model
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

print("\n===== SAMPLE PREDICTIONS =====")

print("First 10 actual grades:")
print(y_test.head(10).values)

print("\nFirst 10 predicted grades:")
print(best_predictions[:10])


# --------------------------------------------------
# Save best model
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/student_performance_model.pkl"
)

print("\nModel saved successfully!")

print(
    f"Saved {best_model_name} "
    "as student_performance_model.pkl"
)
