
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")

FEATURES = [
    "switch", "pktcount", "bytecount", "dur", "dur_nsec",
    "tot_dur", "flows", "packetins", "pktperflow",
    "byteperflow", "pktrate", "Pairflow", "Protocol",
    "port_no", "tx_bytes", "rx_bytes", "tx_kbps",
    "rx_kbps", "tot_kbps"
]


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def safe_match(actual, calculated):
    actual = pd.to_numeric(actual, errors="coerce")
    calculated = pd.to_numeric(calculated, errors="coerce")

    valid = actual.notna() & calculated.notna()

    if valid.sum() == 0:
        return 0, 0, 0.0

    matches = np.isclose(
        actual[valid],
        calculated[valid],
        rtol=0,
        atol=1e-8
    )

    total = int(valid.sum())
    correct = int(matches.sum())

    return correct, total, correct / total * 100


def test_formula(df, name, calculated):

    correct, total, percentage = safe_match(
        df[name],
        calculated
    )

    print(
        f"{name:<15} | {correct:>8}/{total:<8} | "
        f"{percentage:>8.2f}%"
    )


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = FEATURES + ["dt", "src", "dst", "label"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Convert numeric features safely.
    # Protocol remains categorical.
    for feature in FEATURES:

        if feature != "Protocol":
            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    section(
        "DATASET 1 - COMPLETE FEATURE RELATIONSHIP ANALYSIS"
    )

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Unique timestamps:", df["dt"].nunique())
    print("Unique switches:", df["switch"].nunique())
    print("Unique source IPs:", df["src"].nunique())
    print("Unique destination IPs:", df["dst"].nunique())

    numeric_features = [
        feature for feature in FEATURES
        if feature != "Protocol"
    ]

    # --------------------------------------------------
    # 1. FEATURE DISTRIBUTIONS
    # --------------------------------------------------

    section("1. FEATURE DISTRIBUTIONS")

    for feature in FEATURES:

        series = df[feature]

        print(f"\n{feature}")
        print("-" * 60)

        print("Unique:", series.nunique())
        print("Missing:", series.isna().sum())

        if pd.api.types.is_numeric_dtype(series):

            print("Negative:", (series < 0).sum())
            print("Zero:", (series == 0).sum())
            print("Minimum:", series.min())
            print("Maximum:", series.max())
            print("Mean:", round(series.mean(), 4))
            print("Median:", round(series.median(), 4))

        else:

            print("Categorical values:")
            print(
                series.value_counts(dropna=False).to_string()
            )

    # --------------------------------------------------
    # 2. CORRELATION MATRIX
    # --------------------------------------------------

    section("2. NUMERIC FEATURE CORRELATION MATRIX")

    correlation = df[numeric_features].corr()

    print(correlation.round(3).to_string())

    print("\nHIGH CORRELATION PAIRS (ABSOLUTE >= 0.80)")
    print("-" * 60)

    for i, first in enumerate(numeric_features):

        for second in numeric_features[i + 1:]:

            value = correlation.loc[first, second]

            if pd.notna(value) and abs(value) >= 0.80:

                print(
                    f"{first:<15} - {second:<15}: "
                    f"{value:.4f}"
                )

    # --------------------------------------------------
    # 3. MATHEMATICAL RELATIONSHIPS
    # --------------------------------------------------

    section("3. CANDIDATE MATHEMATICAL RELATIONSHIPS")

    print(
        f"{'Feature':<15} | {'Matches':<18} | {'Match %':>8}"
    )
    print("-" * 60)

    candidates = {
        "pktrate":
            np.floor(df["pktperflow"] / 30),

        "byteperflow":
            (
                df["bytecount"]
                / df["pktcount"].replace(0, np.nan)
            ) * df["pktperflow"],

        "tot_kbps":
            df["tx_kbps"] + df["rx_kbps"],

        "tx_kbps":
            (
                df["tx_bytes"] * 8
            ) / (
                df["dur"].replace(0, np.nan) * 1000
            ),

        "rx_kbps":
            (
                df["rx_bytes"] * 8
            ) / (
                df["dur"].replace(0, np.nan) * 1000
            ),

        "tot_dur":
            df["dur"] * 1e9 + df["dur_nsec"],

        "pktperflow":
            df["pktcount"] /
            df["flows"].replace(0, np.nan),

        "byteperflow":
            df["bytecount"] /
            df["flows"].replace(0, np.nan)
    }

    for feature, calculated in candidates.items():

        test_formula(
            df,
            feature,
            calculated
        )

    # --------------------------------------------------
    # 4. GROUP CONSISTENCY
    # --------------------------------------------------

    section("4. FEATURE AGGREGATION ANALYSIS")

    grouping_options = {
        "timestamp + switch":
            ["dt", "switch"],

        "timestamp + switch + protocol":
            ["dt", "switch", "Protocol"],

        "timestamp + switch + protocol + port":
            ["dt", "switch", "Protocol", "port_no"],

        "timestamp + switch + src + dst + protocol":
            ["dt", "switch", "src", "dst", "Protocol"]
    }

    for group_name, group_columns in grouping_options.items():

        print(f"\nGrouping: {group_name}")
        print("-" * 60)

        grouped = df.groupby(
            group_columns,
            dropna=False
        )[FEATURES].nunique()

        for feature in FEATURES:

            consistent = (
                grouped[feature] <= 1
            ).sum()

            total = len(grouped)

            percentage = (
                consistent / total * 100
                if total else 0
            )

            print(
                f"{feature:<15}: "
                f"{consistent}/{total} groups consistent "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------
    # 5. LABEL RELATIONSHIPS
    # --------------------------------------------------

    section("5. FEATURE DISTRIBUTION BY LABEL")

    for feature in FEATURES:

        print(f"\n{feature}")
        print("-" * 60)

        if pd.api.types.is_numeric_dtype(df[feature]):

            print(
                df.groupby("label")[feature]
                .agg([
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max"
                ])
                .round(3)
                .to_string()
            )

        else:

            print(
                pd.crosstab(
                    df["label"],
                    df[feature],
                    normalize="index"
                ).round(4).to_string()
            )

    # --------------------------------------------------
    # 6. DUPLICATE FEATURE COMBINATIONS
    # --------------------------------------------------

    section("6. DUPLICATE FEATURE COMBINATIONS")

    duplicate_count = df.duplicated().sum()

    print("Exact duplicate rows:", duplicate_count)

    feature_duplicates = df.duplicated(
        subset=FEATURES,
        keep=False
    )

    print(
        "Rows sharing identical 19-feature combinations:",
        feature_duplicates.sum()
    )

    # Check if identical feature combinations have
    # conflicting labels.
    label_counts = df.groupby(
        FEATURES,
        dropna=False
    )["label"].nunique()

    conflicting = label_counts[label_counts > 1]

    print(
        "Feature combinations with conflicting labels:",
        len(conflicting)
    )

    # --------------------------------------------------
    # 7. TIMESTAMP PATTERNS
    # --------------------------------------------------

    section("7. TIMESTAMP PATTERNS")

    timestamp_stats = df.groupby("dt").agg(
        records=("dt", "size"),
        switches=("switch", "nunique"),
        pktcount_mean=("pktcount", "mean"),
        packetins_unique=("packetins", "nunique"),
        flows_unique=("flows", "nunique")
    )

    print("\nFirst 15 timestamps:")
    print(
        timestamp_stats.head(15)
        .round(3)
        .to_string()
    )

    print("\nLast 15 timestamps:")
    print(
        timestamp_stats.tail(15)
        .round(3)
        .to_string()
    )

    # --------------------------------------------------
    # 8. SWITCH-LEVEL PATTERNS
    # --------------------------------------------------

    section("8. SWITCH-LEVEL FEATURE PATTERNS")

    switch_stats = df.groupby("switch").agg(
        records=("switch", "size"),
        timestamps=("dt", "nunique"),
        pktcount_mean=("pktcount", "mean"),
        bytecount_mean=("bytecount", "mean"),
        flows_mean=("flows", "mean"),
        packetins_mean=("packetins", "mean"),
        tx_bytes_mean=("tx_bytes", "mean"),
        rx_bytes_mean=("rx_bytes", "mean")
    )

    print(
        switch_stats.round(3).to_string()
    )

    # --------------------------------------------------
    # 9. SOURCE AND DESTINATION PATTERNS
    # --------------------------------------------------

    section("9. SOURCE AND DESTINATION PATTERNS")

    pairs = df.groupby(["src", "dst"]).size()

    print("Unique source-destination pairs:", len(pairs))

    print("\nMost frequent pairs:")
    print(
        pairs.sort_values(ascending=False)
        .head(20)
        .to_string()
    )

    print("\nPairflow distribution by protocol:")

    print(
        pd.crosstab(
            df["Protocol"],
            df["Pairflow"],
            normalize="index"
        ).round(4).to_string()
    )

    # --------------------------------------------------
    # 10. FINAL SUMMARY
    # --------------------------------------------------

    section("10. FINAL SUMMARY")

    print("Records:", len(df))
    print("Numeric features:", len(numeric_features))
    print("Categorical features: Protocol")
    print("Labels:", df["label"].value_counts().to_dict())
    print("Exact duplicates:", duplicate_count)
    print("Conflicting feature combinations:", len(conflicting))

    print("\nFeature relationship investigation completed.")


if __name__ == "__main__":
    main()
