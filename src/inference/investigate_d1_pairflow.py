
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "dt",
        "switch",
        "src",
        "dst",
        "Protocol",
        "port_no",
        "pktcount",
        "bytecount",
        "flows",
        "Pairflow",
        "label"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("=" * 70)
    print("DATASET 1 - PAIRFLOW INVESTIGATION")
    print("=" * 70)

    print("Total records:", len(df))

    # --------------------------------------------------
    # 1. Overall statistics
    # --------------------------------------------------

    print("\n1. OVERALL STATISTICS")
    print("-" * 70)

    print(df["Pairflow"].describe().to_string())

    print("\nUnique Pairflow values:")
    print("Count:", df["Pairflow"].nunique())
    print(
        df["Pairflow"]
        .value_counts()
        .sort_index()
        .head(40)
        .to_string()
    )

    # --------------------------------------------------
    # 2. Value categories
    # --------------------------------------------------

    print("\n2. VALUE CATEGORIES")
    print("-" * 70)

    print("Zero values:", (df["Pairflow"] == 0).sum())
    print("Positive values:", (df["Pairflow"] > 0).sum())
    print("Negative values:", (df["Pairflow"] < 0).sum())

    print(
        "Missing values:",
        df["Pairflow"].isna().sum()
    )

    # --------------------------------------------------
    # 3. Check binary behaviour
    # --------------------------------------------------

    print("\n3. BINARY FEATURE CHECK")
    print("-" * 70)

    unique_values = set(df["Pairflow"].dropna().unique())

    print("Unique values:", sorted(unique_values)[:50])

    if unique_values.issubset({0, 1}):
        print("Pairflow behaves as a binary feature.")
    else:
        print("Pairflow is not strictly binary.")

    # --------------------------------------------------
    # 4. Distribution by protocol
    # --------------------------------------------------

    print("\n4. DISTRIBUTION BY PROTOCOL")
    print("-" * 70)

    print(
        df.groupby("Protocol")["Pairflow"]
        .agg(["count", "nunique", "min", "max", "mean", "median", "std"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 5. Distribution by switch
    # --------------------------------------------------

    print("\n5. DISTRIBUTION BY SWITCH")
    print("-" * 70)

    print(
        df.groupby("switch")["Pairflow"]
        .agg(["count", "nunique", "min", "max", "mean", "median"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 6. Correlation with other features
    # --------------------------------------------------

    print("\n6. CORRELATION WITH OTHER FEATURES")
    print("-" * 70)

    numeric_features = [
        "pktcount",
        "bytecount",
        "flows",
        "port_no",
        "switch"
    ]

    for feature in numeric_features:
        correlation = df["Pairflow"].corr(df[feature])

        print(
            f"Correlation with {feature}: "
            f"{correlation:.6f}"
        )

    # --------------------------------------------------
    # 7. Group consistency
    # --------------------------------------------------

    print("\n7. GROUP CONSISTENCY")
    print("-" * 70)

    grouping_options = [
        ["dt", "switch"],
        ["dt", "switch", "Protocol"],
        ["dt", "switch", "Protocol", "port_no"],
        ["dt", "switch", "src", "dst", "Protocol"]
    ]

    for group_columns in grouping_options:

        grouped = df.groupby(
            group_columns,
            dropna=False
        )["Pairflow"].agg(
            records="size",
            unique_values="nunique",
            minimum="min",
            maximum="max"
        )

        inconsistent = grouped[
            grouped["unique_values"] > 1
        ]

        print(f"\nGrouping by: {group_columns}")
        print("Total groups:", len(grouped))
        print("Inconsistent groups:", len(inconsistent))

        if not inconsistent.empty:
            print("\nFirst 10 inconsistent groups:")
            print(
                inconsistent.head(10).to_string()
            )

    # --------------------------------------------------
    # 8. Relationship with attack label
    # --------------------------------------------------

    print("\n8. PAIRFLOW BY LABEL")
    print("-" * 70)

    print(
        df.groupby("label")["Pairflow"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 9. Duplicate and repeated values
    # --------------------------------------------------

    print("\n9. REPEATED VALUE ANALYSIS")
    print("-" * 70)

    repeated = df["Pairflow"].value_counts()

    print("Values occurring more than once:")
    print((repeated > 1).sum())

    print("\nTop 20 most frequent Pairflow values:")
    print(repeated.head(20).to_string())

    # --------------------------------------------------
    # 10. Sample records
    # --------------------------------------------------

    print("\n10. SAMPLE RECORDS")
    print("-" * 70)

    sample_columns = [
        "dt",
        "switch",
        "src",
        "dst",
        "Protocol",
        "port_no",
        "pktcount",
        "bytecount",
        "flows",
        "Pairflow",
        "label"
    ]

    print(
        df[sample_columns]
        .sample(min(30, len(df)), random_state=42)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 11. Final summary
    # --------------------------------------------------

    print("\n11. FINAL SUMMARY")
    print("=" * 70)

    print("Total records:", len(df))
    print("Unique Pairflow values:", df["Pairflow"].nunique())
    print("Zero values:", (df["Pairflow"] == 0).sum())
    print("Positive values:", (df["Pairflow"] > 0).sum())
    print("Negative values:", (df["Pairflow"] < 0).sum())

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
