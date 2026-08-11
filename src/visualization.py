import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the student performance dataset."""
    return pd.read_csv(file_path, sep=";")


if __name__ == "__main__":
    file_path = "data/raw/student-mat.csv"

    data = load_data(file_path)

    print("\n===== STUDY TIME VS FINAL GRADE =====")

    studytime_performance = data.groupby("studytime")["G3"].mean()

    print(studytime_performance)

    print("\n===== FAILURES VS FINAL GRADE =====")

    failure_performance = data.groupby("failures")["G3"].mean()

    print(failure_performance)

    plt.figure(figsize=(8, 5))

    failure_performance.plot(kind="bar", edgecolor="black")

    plt.title("Final Grade by Previous Failures")
    plt.xlabel("Number of Previous Failures")
    plt.ylabel("Average Final Grade")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "reports/figures/failures_vs_grade.png",
        dpi=300
    )

    print("Failure visualization saved successfully!")


    plt.figure(figsize=(8, 5))

    data.boxplot(column="G3", by="studytime")

    plt.title("Final Grade by Study Time")
    plt.suptitle("")
    plt.xlabel("Study Time")
    plt.ylabel("Final Grade (G3)")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/studytime_vs_grade.png",
        dpi=300
    )

    print("Study time visualization saved successfully!")

    plt.show()
