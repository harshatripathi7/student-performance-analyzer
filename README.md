# 🎓 Student Performance Analyzer

A machine learning project that analyzes student academic performance and predicts a student's final grade based on academic, demographic, and lifestyle-related factors.

The project includes exploratory data analysis, data visualization, machine learning model training, performance evaluation, and an interactive Streamlit web application.

---

## 🚀 Project Demo

The application allows users to enter student information and receive a predicted final grade out of 20.

### Example

**Input**

- Age: 17
- Study Time: 3
- Previous Failures: 0
- Absences: 4
- G1: 12
- G2: 13
- Mother's Education: 3
- Father's Education: 2
- Free Time: 3
- Going Out: 3
- Health: 4

**Prediction**

> Predicted Final Grade: ~12.9 / 20

---

## 📌 Objectives

The main objectives of this project are:

- Analyze factors associated with student academic performance.
- Perform exploratory data analysis on student records.
- Visualize relationships between study habits and final grades.
- Train a machine learning regression model.
- Evaluate model performance using standard regression metrics.
- Build an interactive web application for grade prediction.

---

## 📊 Dataset

The project uses the **Student Performance Dataset**, containing information about students' demographic characteristics, family background, study habits, lifestyle, previous grades, absences, and final grades.

The dataset contains:

- **395 students**
- **33 features**

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

The target variable is:

```text
G3 — Final Grade

🔍 Exploratory Data Analysis

The dataset was analyzed to understand the distribution of final grades and the relationship between different student characteristics and academic performance.

Final Grade Distribution

The average final grade is approximately:

10.42 / 20

The median final grade is:

11 / 20

The highest recorded grade is:

20 / 20

The lowest recorded grade is:

0 / 20
📈 Key Findings
Study Time vs Final Grade

Students with higher study-time levels generally achieved higher average final grades.

Study Time	Average G3
1	10.05
2	10.17
3	11.40
4	11.26
Previous Failures vs Final Grade

Previous academic failures showed a strong relationship with final performance.

Previous Failures	Average G3
0	11.25
1	8.12
2	6.24
3	5.69

This suggests that students with more previous failures tend to have lower final grades in this dataset.

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
📏 Model Performance

The Linear Regression model achieved:

Metric	Score
MAE	1.38
RMSE	2.16
R² Score	0.77
Interpretation

An R² score of 0.77 means that the model explains approximately 77% of the variance in final grades within the test dataset.

The MAE of 1.38 means that the model's predictions are, on average, about 1.38 grade points away from the actual final grade.

🌐 Streamlit Web Application

The project includes an interactive web application built using Streamlit.

Users can enter:

Student age
Study time
Previous failures
Absences
First period grade
Second period grade
Parents' education levels
Free time
Going-out frequency
Health level

The application then predicts the student's final grade.

Run the application
streamlit run app/app.py

The application will be available locally at:

http://localhost:8501
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
📁 Project Structure
student-performance-analyzer/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
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
├── requirements.txt
└── LICENSE
⚙️ Installation

Clone the repository:

git clone <YOUR-GITHUB-REPOSITORY-URL>

Move into the project directory:

cd student-performance-analyzer

Create a virtual environment:

python3 -m venv .venv

Activate it:

macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
Run data analysis
python3 src/eda.py
Generate visualizations
python3 src/visualization.py
Train the model
python3 src/train_model.py
Make a prediction from the terminal
python3 src/predict.py
Launch the web application
streamlit run app/app.py
🔮 Future Improvements

Potential improvements include:

Compare Linear Regression with Random Forest and Gradient Boosting.
Add hyperparameter tuning.
Add cross-validation.
Add feature importance analysis.
Add prediction confidence/error analysis.
Add interactive visualizations to the Streamlit application.
Add model comparison charts.
Deploy the application online.
Add automated unit tests.
Add CI/CD using GitHub Actions.
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

This project is being expanded with additional machine learning models, improved visualizations, testing, and deployment.


Save:

**Control + O → Enter → Control + X**

---

# STEP 16.2 — Check the README

Run:

```bash
cat README.md

