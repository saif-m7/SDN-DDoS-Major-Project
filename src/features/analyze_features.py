from pathlib import Path

import pandas as pd
import numpy as np


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


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_features(df, dataset_name, target_column="label"):

    print("\n" + "=" * 80)
    print(f"FEATURE ANALYSIS: {dataset_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    X = df[feature_columns]
    y = df[target_column]

    print("\nTotal features:", len(feature_columns))

    # --------------------------------------------------------
    # Numerical and categorical features
    # --------------------------------------------------------

    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("\n--- NUMERICAL FEATURES ---")

    for feature in numerical_features:
        print(feature)

    print("\n--- CATEGORICAL FEATURES ---")

    for feature in categorical_features:
        print(feature)

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    print("\n--- UNIQUE VALUE ANALYSIS ---")

    unique_data = []

    for feature in feature_columns:

        unique_count = X[feature].nunique(
            dropna=False
        )

        unique_data.append(
            {
                "feature": feature,
                "unique_values": unique_count,
                "data_type": str(X[feature].dtype)
            }
        )

    unique_df = pd.DataFrame(unique_data)

    print(
        unique_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    print("\n--- CONSTANT FEATURES ---")

    constant_features = [
        feature
        for feature in feature_columns
        if X[feature].nunique(dropna=False) <= 1
    ]

    if constant_features:

        for feature in constant_features:
            print(feature)

    else:
        print("No constant features.")

    # --------------------------------------------------------
    # Near-constant features
    # --------------------------------------------------------

    print("\n--- LOW-VARIANCE FEATURES ---")

    for feature in numerical_features:

        value_counts = (
            X[feature]
            .value_counts(normalize=True)
        )

        if not value_counts.empty:

            dominant_percentage = (
                value_counts.iloc[0] * 100
            )

            if dominant_percentage >= 99:

                print(
                    f"{feature}: "
                    f"{dominant_percentage:.2f}% "
                    f"same value"
                )

    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    print("\n--- NEGATIVE VALUE CHECK ---")

    for feature in numerical_features:

        negative_count = (
            X[feature] < 0
        ).sum()

        if negative_count > 0:

            percentage = (
                negative_count
                / len(X)
                * 100
            )

            print(
                f"{feature}: "
                f"{negative_count:,} "
                f"negative values "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    print("\n--- INFINITE VALUE CHECK ---")

    infinite_features = []

    for feature in numerical_features:

        infinite_count = np.isinf(
            X[feature]
        ).sum()

        if infinite_count > 0:

            infinite_features.append(
                (feature, infinite_count)
            )

    if infinite_features:

        for feature, count in infinite_features:
            print(
                f"{feature}: "
                f"{count:,}"
            )

    else:
        print("No infinite values found.")

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print("\n--- HIGH CORRELATION PAIRS ---")

    if len(numerical_features) > 1:

        correlation_matrix = (
            X[numerical_features]
            .corr()
            .abs()
        )

        high_correlation_pairs = []

        for i in range(
            len(correlation_matrix.columns)
        ):

            for j in range(i + 1,
                           len(correlation_matrix.columns)):

                feature_1 = (
                    correlation_matrix.columns[i]
                )

                feature_2 = (
                    correlation_matrix.columns[j]
                )

                correlation = (
                    correlation_matrix.iloc[i, j]
                )

                if correlation >= 0.90:

                    high_correlation_pairs.append(
                        (
                            feature_1,
                            feature_2,
                            correlation
                        )
                    )

        if high_correlation_pairs:

            for feature_1, feature_2, correlation in (
                high_correlation_pairs
            ):

                print(
                    f"{feature_1} <-> "
                    f"{feature_2} : "
                    f"{correlation:.4f}"
                )

        else:
            print(
                "No feature pairs with "
                "absolute correlation >= 0.90."
            )

    # --------------------------------------------------------
    # Correlation with target
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(y):

        print(
            "\n--- CORRELATION WITH NUMERICAL TARGET ---"
        )

        correlation_with_target = (
            X[numerical_features]
            .corrwith(y)
            .sort_values(
                key=lambda values:
                values.abs(),
                ascending=False
            )
        )

        print(
            correlation_with_target
        )

    else:

        print(
            "\n--- TARGET IS CATEGORICAL ---"
        )

        print(
            "Target correlation will be "
            "handled after encoding."
        )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    print(
        "\n--- NUMERICAL FEATURE STATISTICS ---"
    )

    print(
        X[numerical_features]
        .describe()
        .transpose()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading Dataset 1...")

    dataset_1 = pd.read_csv(DATASET_1)

    analyze_features(
        dataset_1,
        "Dataset 1 - SDN DDoS Detection"
    )

    print("\n\nLoading Dataset 2 Training Data...")

    dataset_2_train = pd.read_csv(
        DATASET_2_TRAIN
    )

    analyze_features(
        dataset_2_train,
        "Dataset 2 - DDoS Classification"
    )

    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 80)