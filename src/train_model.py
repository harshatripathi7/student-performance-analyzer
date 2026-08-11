import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "data/raw/student-mat.csv"

data = pd.read_csv(file_path, sep=";")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# Select features and target
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
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== DATA SPLIT =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Train Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

print("\n===== MODEL TRAINED =====")
print("Linear Regression model trained successfully!")

# Make predictions on test data
y_pred = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print("First 10 actual grades:")
print(y_test.head(10).values)

print("\nFirst 10 predicted grades:")
print(y_pred[:10])
# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
# Save trained model
joblib.dump(model, "models/student_performance_model.pkl")

print("\nModel saved successfully!")
