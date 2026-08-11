import pandas as pd


def load_data(file_path):
    """Load the student performance dataset."""
    return pd.read_csv(file_path, sep=";")


if __name__ == "__main__":
    file_path = "data/raw/student-mat.csv"

    data = load_data(file_path)

    print("===== DATASET OVERVIEW =====")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\n===== MISSING VALUES =====")
    print(data.isnull().sum())

    print("\n===== DATA TYPES =====")
    print(data.dtypes)

    print("\n===== NUMERICAL SUMMARY =====")
    print(data.describe())

    print("\n===== FINAL GRADE INFORMATION =====")
    print(f"Average final grade: {data['G3'].mean():.2f}")
    print(f"Highest final grade: {data['G3'].max()}")
    print(f"Lowest final grade: {data['G3'].min()}")
    print(f"Median final grade: {data['G3'].median():.2f}")