import pandas as pd


def load_data(file_path):
    """Load student data from a CSV file."""
    data = pd.read_csv(file_path, sep=";")
    return data


if __name__ == "__main__":
    file_path = "data/raw/student-mat.csv"

    data = load_data(file_path)

    print("Dataset loaded successfully!")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nDataset information:")
    print(data.info())

    print("\nStatistical summary:")
    print(data.describe())