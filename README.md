# 🎓 Student Performance Analyzer

A machine learning project that analyzes student academic performance and predicts a student's final grade based on academic, demographic, and lifestyle-related factors.

The project covers the complete machine learning workflow:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Multiple regression models
- Model comparison
- Model evaluation
- Prediction system
- Interactive Streamlit dashboard
- Error and residual analysis
- GitHub version control

---

## 🚀 Project Overview

The goal of this project is to predict a student's final grade (`G3`) out of 20 using information such as:

- Previous grades
- Study time
- Previous failures
- Absences
- Parents' education
- Free time
- Social activity
- Health
- Age

The project compares three machine learning regression algorithms:

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression

The **Random Forest model** achieved the best performance.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit dashboard where users can enter student information and receive a predicted final grade.

### Application Features

- 🎓 Student information input
- 🔮 Final grade prediction
- 📈 Estimated percentage
- 💡 Personalized recommendations
- 📊 Model performance metrics
- 🎯 Actual vs Predicted analysis
- 📉 Residual analysis
- 📊 Prediction error distribution
- 🔍 Exploratory data visualizations

### Run the application

```bash
streamlit run app/app.py

The application will open at:

http://localhost:8501
📊 Dataset

The project uses the Student Performance Dataset containing academic, demographic, family, and lifestyle information about students.

Dataset Statistics
Property	Value
Students	395
Features	33
Target	G3
Target Range	0–20

The target variable is:

G3 — Final Grade
🔑 Important Features
Feature	Description
age	Student age
studytime	Weekly study time
failures	Number of previous class failures
absences	Number of school absences
G1	First period grade
G2	Second period grade
Medu	Mother's education level
Fedu	Father's education level
freetime	Free time after school
goout	Frequency of going out
health	Current health status
🔍 Exploratory Data Analysis

The dataset was analyzed to understand the distribution of student grades and identify relationships between academic performance and different student characteristics.

Final Grade Distribution

The dataset contains grades ranging from:

0 → 20

Average final grade:

≈ 10.42 / 20

Median final grade:

11 / 20

Highest recorded grade:

20 / 20

Lowest recorded grade:

0 / 20
Visualization

📚 Study Time vs Final Grade

Students with higher study-time levels generally achieved higher average final grades.

Study Time	Average G3
1	10.05
2	10.17
3	11.40
4	11.26

❌ Previous Failures vs Final Grade

Previous academic failures showed a strong relationship with final performance.

Previous Failures	Average G3
0	11.25
1	8.12
2	6.24
3	5.69

Students with more previous failures tended to have lower final grades in this dataset.

🤖 Machine Learning

Three regression models were trained and compared.

Selected Features
age
studytime
failures
absences
G1
G2
Medu
Fedu
freetime
goout
health
Dataset Split

The dataset was divided into:

Dataset	Samples
Training	316
Testing	79
Total	395

The split was performed using:

train_test_split(
    test_size=0.2,
    random_state=42
)
🏆 Model Comparison

Three machine learning algorithms were evaluated.

Model	MAE	RMSE	R²
Linear Regression	1.38	2.16	0.77
Random Forest	1.08	1.78	0.85
Gradient Boosting	1.16	1.85	0.83
🥇 Best Model: Random Forest

The Random Forest model achieved the best overall performance.

Performance

MAE

1.08

RMSE

1.78

R² Score

0.85

An R² score of 0.85 means that the model explains approximately 85% of the variance in final grades on the test dataset.

The MAE of 1.08 means that predictions differ from actual grades by approximately 1.08 grade points on average.

📈 Model Evaluation

The Random Forest model was evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
Actual vs Predicted analysis
Residual analysis
Error distribution
🎯 Actual vs Predicted Grades

The actual-vs-predicted plot helps visualize how closely the model's predictions follow the real final grades.

📉 Residual Analysis

Residual analysis helps identify systematic prediction errors and determine whether the model's errors are reasonably distributed.

📊 Prediction Error Distribution

The error distribution provides insight into how frequently the model overestimates or underestimates student performance.

🧠 Machine Learning Workflow

The overall workflow used in this project is:

Raw Dataset
     ↓
Data Loading
     ↓
Data Exploration
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Linear Regression
     ↓
Random Forest
     ↓
Gradient Boosting
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Model Evaluation
     ↓
Prediction
     ↓
Streamlit Dashboard
📁 Project Structure
student-performance-analyzer/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── student-mat.csv
│       └── student.txt
│
├── models/
│   └── student_performance_model.pkl
│
├── reports/
│   └── figures/
│       ├── final_grade_distribution.png
│       ├── studytime_vs_grade.png
│       ├── failures_vs_grade.png
│       ├── actual_vs_predicted.png
│       ├── residual_plot.png
│       └── error_distribution.png
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── visualization.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluate_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
🛠️ Technologies Used
Programming
Python
Data Analysis
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Joblib
Web Application
Streamlit
Version Control
Git
GitHub
⚙️ Installation

Clone the repository:

git clone https://github.com/harshatripathi7/student-performance-analyzer.git

Move into the project directory:

cd student-performance-analyzer

Create a virtual environment:

python3 -m venv .venv

Activate the environment on macOS/Linux:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
Run Exploratory Data Analysis
python3 src/eda.py
Generate Visualizations
python3 src/visualization.py
Train the Models
python3 src/train_model.py
Evaluate the Model
python3 src/evaluate_model.py
Make a Terminal Prediction
python3 src/predict.py
Launch the Streamlit Application
streamlit run app/app.py
🔮 Future Improvements

Potential improvements include:

 Hyperparameter tuning
 Cross-validation
 Feature importance visualization
 SHAP-based explainability
 Prediction confidence analysis
 Interactive Plotly visualizations
 Additional machine learning algorithms
 Automated unit tests
 CI/CD using GitHub Actions
 Online deployment
 Model monitoring
 Improved recommendation system
📌 Key Takeaways

The project demonstrates a complete end-to-end machine learning workflow.

The main findings are:

Previous grades (G1 and G2) are important predictors of final performance.
Previous failures are strongly associated with lower final grades.
Study time shows a positive relationship with academic performance.
Random Forest outperformed Linear Regression and Gradient Boosting on the test dataset.
The final Random Forest model achieved an R² score of 0.85.
👩‍💻 Author
Harsha Tripathi

B.Tech Computer Science & Engineering

Interests
Machine Learning
Artificial Intelligence
Data Science
Software Development
Research
⭐ Project Status

🚧 Actively Developing

This project is being expanded with additional machine learning techniques, explainability, testing, and deployment capabilities.

📄 License

This project is intended for educational and portfolio purposes.



