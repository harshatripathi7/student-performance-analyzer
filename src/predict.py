import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/student_performance_model.pkl")

print("Model loaded successfully!")
print("\n===== STUDENT PERFORMANCE PREDICTOR =====")

# Get student information
age = int(input("Enter student's age (15-22): "))
studytime = int(input("Enter study time (1-4): "))
failures = int(input("Enter previous failures (0-3): "))
absences = int(input("Enter number of absences: "))
G1 = int(input("Enter first period grade (0-20): "))
G2 = int(input("Enter second period grade (0-20): "))
Medu = int(input("Enter mother's education level (0-4): "))
Fedu = int(input("Enter father's education level (0-4): "))
freetime = int(input("Enter free time level (1-5): "))
goout = int(input("Enter going-out level (1-5): "))
health = int(input("Enter health level (1-5): "))

# Create student data
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

# Make prediction
prediction = model.predict(student_data)[0]

print("\n===== PREDICTION RESULT =====")
print(f"Predicted final grade: {prediction:.2f} / 20")
