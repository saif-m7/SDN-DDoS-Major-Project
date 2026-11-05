
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
        "packetins",
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
    print("DATASET 1 - PACKETINS INVESTIGATION")
    print("=" * 70)

    print("Total records:", len(df))

    # --------------------------------------------------
    # 1. Overall statistics
    # --------------------------------------------------

    print("\n1. OVERALL STATISTICS")
    print("-" * 70)

    print(df["packetins"].describe().to_string())

    print("\nUnique packetins values:", df["packetins"].nunique())

    print("\nMost frequent values:")
    print(
        df["packetins"]
        .value_counts()
        .head(20)
        .to_string()
    )

    # --------------------------------------------------
    # 2. Zero, positive and negative values
    # --------------------------------------------------

    print("\n2. VALUE CATEGORIES")
    print("-" * 70)

    print("Zero:", (df["packetins"] == 0).sum())
    print("Positive:", (df["packetins"] > 0).sum())
    print("Negative:", (df["packetins"] < 0).sum())
    print("Missing:", df["packetins"].isna().sum())

    # --------------------------------------------------
    # 3. Distribution by protocol
    # --------------------------------------------------

    print("\n3. DISTRIBUTION BY PROTOCOL")
    print("-" * 70)

    print(
        df.groupby("Protocol")["packetins"]
        .agg(["count", "nunique", "min", "max", "mean", "median", "std"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 4. Distribution by switch
    # --------------------------------------------------

    print("\n4. DISTRIBUTION BY SWITCH")
    print("-" * 70)

    print(
        df.groupby("switch")["packetins"]
        .agg(["count", "nunique", "min", "max", "mean", "median"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 5. Correlation with other features
    # --------------------------------------------------

    print("\n5. CORRELATION WITH OTHER FEATURES")
    print("-" * 70)

    numeric_features = [
        "pktcount",
        "bytecount",
        "flows",
        "port_no",
        "switch"
    ]

    for feature in numeric_features:
        correlation = df["packetins"].corr(df[feature])

        print(
            f"Correlation with {feature}: {correlation:.6f}"
        )

    # --------------------------------------------------
    # 6. Group consistency
    # --------------------------------------------------

    print("\n6. GROUP CONSISTENCY")
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
        )["packetins"].agg(
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
    # 7. Relationship with timestamp
    # --------------------------------------------------

    print("\n7. TIMESTAMP ANALYSIS")
    print("-" * 70)

    timestamp_stats = df.groupby("dt")["packetins"].agg(
        count="size",
        mean="mean",
        minimum="min",
        maximum="max",
        total="sum"
    )

    print("\nFirst 15 timestamps:")
    print(timestamp_stats.head(15).round(3).to_string())

    print("\nLast 15 timestamps:")
    print(timestamp_stats.tail(15).round(3).to_string())

    print(
        "\nUnique timestamps:",
        df["dt"].nunique()
    )

    # --------------------------------------------------
    # 8. Relationship with label
    # --------------------------------------------------

    print("\n8. PACKETINS BY LABEL")
    print("-" * 70)

    print(
        df.groupby("label")["packetins"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 9. Candidate formula tests
    # --------------------------------------------------

    print("\n9. CANDIDATE FORMULA TESTS")
    print("-" * 70)

    candidates = {
        "pktcount / flows":
            df["pktcount"] /
            df["flows"].replace(0, np.nan),

        "pktcount / 30":
            df["pktcount"] / 30,

        "bytecount / flows":
            df["bytecount"] /
            df["flows"].replace(0, np.nan),

        "flows * 30":
            df["flows"] * 30
    }

    for name, calculated in candidates.items():

        valid = (
            df["packetins"].notna()
            & calculated.notna()
        )

        matches = np.isclose(
            df.loc[valid, "packetins"],
            calculated[valid],
            rtol=0,
            atol=1e-8
        )

        total = valid.sum()
        correct = matches.sum()

        percentage = (
            correct / total * 100
            if total else 0
        )

        print(
            f"{name}: {correct}/{total} "
            f"({percentage:.2f}%)"
        )

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
        "packetins",
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
    print("Unique packetins values:", df["packetins"].nunique())
    print("Zero values:", (df["packetins"] == 0).sum())
    print("Positive values:", (df["packetins"] > 0).sum())
    print("Negative values:", (df["packetins"] < 0).sum())

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
