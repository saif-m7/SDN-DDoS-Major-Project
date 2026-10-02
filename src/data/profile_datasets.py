from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_1 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset1"
    / "SDN-DDoS_Traffic_Dataset.csv"
)

DATASET_2_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset2"
    / "TRAIN-DATA.csv"
)

DATASET_2_TEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset2"
    / "TEST-DATA.csv"
)


# ============================================================
# DATASET PROFILING FUNCTION
# ============================================================

def profile_dataset(file_path, dataset_name):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print(f"\nFile: {file_path}")

    if not file_path.exists():
        print("ERROR: Dataset file not found.")
        return

    # Load dataset
    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n--- BASIC INFORMATION ---")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    print("\n--- COLUMN NAMES ---")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n--- MISSING VALUES ---")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    # --------------------------------------------------------
    # Duplicate records
    # --------------------------------------------------------

    print("\n--- DUPLICATE RECORDS ---")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    # --------------------------------------------------------
    # Constant columns
    # --------------------------------------------------------

    print("\n--- CONSTANT COLUMNS ---")

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        for column in constant_columns:
            print(
                f"{column} → "
                f"{df[column].iloc[0]}"
            )
    else:
        print("No constant columns found.")

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    print("\n--- UNIQUE VALUES ---")

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        print(
            f"{column:25} : "
            f"{unique_count:,} unique values"
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    if "label" in df.columns:

        print("\n--- LABEL DISTRIBUTION ---")

        label_counts = df["label"].value_counts(dropna=False)

        print(label_counts)

        print("\n--- LABEL PERCENTAGE ---")

        label_percentage = (
            df["label"]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
        )

        print(label_percentage)

    # --------------------------------------------------------
    # Numerical statistics
    # --------------------------------------------------------

    print("\n--- NUMERICAL STATISTICS ---")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:
        print(
            df[numeric_columns]
            .describe()
            .transpose()
        )
    else:
        print("No numerical columns found.")

    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    print("\n--- FIRST 5 RECORDS ---")
    print(df.head())

    print("\n--- PROFILE COMPLETED ---")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    profile_dataset(
        DATASET_1,
        "Dataset 1 - SDN DDoS Traffic"
    )

    profile_dataset(
        DATASET_2_TRAIN,
        "Dataset 2 - Training Data"
    )

    profile_dataset(
        DATASET_2_TEST,
        "Dataset 2 - Test Data"
    )

    print("\n" + "=" * 70)
    print("ALL DATASETS PROFILED SUCCESSFULLY")
    print("=" * 70)