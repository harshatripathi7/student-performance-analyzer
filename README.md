# 🎓 Student Performance Analyzer

A machine learning project that analyzes student academic performance and predicts a student's final grade based on academic, demographic, and lifestyle-related factors.

The project covers the complete machine learning workflow:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature selection
- Machine learning model training
- Model evaluation
- Saved model inference
- Interactive Streamlit application

---

## 🚀 Project Overview

The goal of this project is to understand which factors are associated with student academic performance and build a regression model capable of predicting a student's final grade (`G3`) out of 20.

The trained Linear Regression model achieved:

| Metric | Score |
|---|---:|
| MAE | **1.38** |
| RMSE | **2.16** |
| R² Score | **0.77** |

The project also includes an interactive Streamlit application where users can enter student information and receive a predicted final grade.

---

## 📊 Dataset

The project uses the **Student Performance Dataset**, containing information about students' demographics, family background, study habits, lifestyle, previous grades, absences, and final grades.

### Dataset characteristics

- **395 students**
- **33 columns**
- **16 numerical features**
- **17 categorical features**
- Target variable: `G3`

### Important Features

| Feature | Description |
|---|---|
| `age` | Student age |
| `studytime` | Weekly study time |
| `failures` | Number of previous class failures |
| `absences` | Number of school absences |
| `G1` | First period grade |
| `G2` | Second period grade |
| `G3` | Final grade |
| `Medu` | Mother's education level |
| `Fedu` | Father's education level |
| `freetime` | Free time after school |
| `goout` | Frequency of going out |
| `health` | Current health status |

### Target

```text
G3 — Final Grade

The final grade ranges from 0 to 20.

🔍 Exploratory Data Analysis

The dataset was analyzed to understand grade distributions and relationships between student characteristics and academic performance.

Final Grade Statistics
Statistic	Value
Average	10.42
Median	11.00
Highest	20
Lowest	0
📈 Visualizations
Final Grade Distribution

Study Time vs Final Grade

Previous Failures vs Final Grade

🔎 Key Findings
Study Time vs Final Grade

Average final grade by study-time level:

Study Time	Average G3
1	10.05
2	10.17
3	11.40
4	11.26

Students with higher study-time levels generally achieved somewhat higher average final grades in this dataset.

However, the relationship is not strictly linear, indicating that study time alone does not determine academic performance.

Previous Failures vs Final Grade
Previous Failures	Average G3
0	11.25
1	8.12
2	6.24
3	5.69

Previous academic failures show a strong negative relationship with final performance in this dataset.

Students with more previous failures tend to have substantially lower average final grades.

🤖 Machine Learning

A Linear Regression model was trained to predict the final grade.

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
Total samples:       395
Training samples:    316
Testing samples:      79

The data was split into training and testing sets before model training.

📏 Model Performance

The Linear Regression model achieved the following results on the test set:

Metric	Score
MAE	1.38
RMSE	2.16
R²	0.77
Interpretation

MAE = 1.38

On average, the model's prediction differs from the actual final grade by approximately 1.38 grade points.

RMSE = 2.16

RMSE penalizes larger prediction errors more strongly than MAE. The value indicates that larger prediction errors are present in the test set.

R² = 0.77

The model explains approximately 77% of the variance in final grades on the test dataset.

Note: These results are specific to the current train/test split and should not be interpreted as proof that the model will achieve the same performance on unseen datasets.

🌐 Streamlit Application

The project includes an interactive web application built with Streamlit.

Users can enter:

Student age
Study time
Previous failures
Absences
First period grade
Second period grade
Mother's education
Father's education
Free time
Going-out frequency
Health level

The application then predicts the student's final grade.

Example

Input

Age: 17
Study Time: 3
Previous Failures: 0
Absences: 4
G1: 12
G2: 13
Mother's Education: 3
Father's Education: 2
Free Time: 3
Going Out: 3
Health: 4

Prediction

Predicted Final Grade: ~12.9 / 20
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Jupyter Notebook
Git & GitHub
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
├── notebooks/
│
├── reports/
│   └── figures/
│       ├── final_grade_distribution.png
│       ├── studytime_vs_grade.png
│       └── failures_vs_grade.png
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── visualization.py
│   ├── train_model.py
│   └── predict.py
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
⚙️ Installation
1. Clone the repository
git clone git@github.com:harshatripathi7/student-performance-analyzer.git
cd student-performance-analyzer
2. Create a virtual environment
python3 -m venv .venv
3. Activate the environment
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
▶️ Running the Project
Run EDA
python3 src/eda.py
Generate visualizations
python3 src/visualization.py
Train the model
python3 src/train_model.py
Make a terminal prediction
python3 src/predict.py
Launch the Streamlit application
streamlit run app/app.py

The application will be available at:

http://localhost:8501
🔮 Future Improvements

Planned improvements include:

Compare Linear Regression with Random Forest and Gradient Boosting
Perform cross-validation
Add hyperparameter tuning
Add feature importance analysis
Add prediction error analysis
Add interactive EDA charts to the Streamlit dashboard
Add model comparison visualizations
Improve application UI/UX
Deploy the application online
Add automated unit tests
Add GitHub Actions for CI/CD
👩‍💻 Author

Harsha Tripathi

B.Tech Computer Science & Engineering

Interested in:

Machine Learning
Artificial Intelligence
Data Science
Software Development
Research
⭐ Project Status

🚧 Actively developing

The current version includes EDA, visualization, Linear Regression, model evaluation, saved-model inference, and an interactive Streamlit application.

Future versions will focus on model comparison, explainability, testing, and deployment.



