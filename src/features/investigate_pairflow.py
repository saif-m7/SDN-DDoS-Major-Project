import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/raw/dataset1/dataset_sdn.csv")


def main():

    print("=" * 70)
    print("PAIRFLOW INVESTIGATION - DATASET 1")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load raw dataset
    # ---------------------------------------------------------
    df = pd.read_csv(RAW_PATH)

    print(f"\nRaw dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Remove exact duplicates
    # ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().copy()

    print(f"Rows before duplicate removal: {before}")
    print(f"Rows after duplicate removal : {len(df)}")
    print(f"Duplicates removed            : {before - len(df)}")

    # ---------------------------------------------------------
    # 3. Convert dt to numeric
    # ---------------------------------------------------------
    df["dt"] = pd.to_numeric(df["dt"], errors="coerce")

    # Sort chronologically
    df = df.sort_values("dt").reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Basic Pairflow statistics
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("OVERALL PAIRFLOW STATISTICS")
    print("-" * 70)

    print(df["Pairflow"].describe())

    print("\nUnique Pairflow values:", df["Pairflow"].nunique())

    # ---------------------------------------------------------
    # 5. Pairflow statistics by label
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PAIRFLOW BY LABEL")
    print("-" * 70)

    label_stats = (
        df.groupby("label")["Pairflow"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max",
            "nunique"
        ])
    )

    print(label_stats)

    # ---------------------------------------------------------
    # 6. Pairflow statistics by temporal period
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PAIRFLOW BY TEMPORAL PERIOD")
    print("-" * 70)

    unique_dt = np.sort(df["dt"].dropna().unique())

    n = len(unique_dt)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_dt = unique_dt[:train_end]
    val_dt = unique_dt[train_end:val_end]
    test_dt = unique_dt[val_end:]

    periods = {
        "TRAIN": train_dt,
        "VALIDATION": val_dt,
        "TEST": test_dt
    }

    for name, dt_values in periods.items():

        subset = df[df["dt"].isin(dt_values)]

        print(f"\n{name}")

        print(f"Rows       : {len(subset)}")
        print(f"dt range   : {subset['dt'].min()} -> {subset['dt'].max()}")
        print(f"Pairflow mean   : {subset['Pairflow'].mean():.6f}")
        print(f"Pairflow std    : {subset['Pairflow'].std():.6f}")
        print(f"Pairflow min    : {subset['Pairflow'].min():.6f}")
        print(f"Pairflow max    : {subset['Pairflow'].max():.6f}")
        print(f"Unique values   : {subset['Pairflow'].nunique()}")

    # ---------------------------------------------------------
    # 7. Pairflow distribution by time
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PAIRFLOW OVER TIME")
    print("-" * 70)

    time_stats = (
        df.groupby("dt")["Pairflow"]
        .agg(["count", "mean", "std", "min", "max", "nunique"])
        .reset_index()
    )

    print("\nFirst 10 time points:")
    print(time_stats.head(10).to_string(index=False))

    print("\nLast 10 time points:")
    print(time_stats.tail(10).to_string(index=False))

    # ---------------------------------------------------------
    # 8. Check whether Pairflow becomes constant
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("CONSTANT-VALUE CHECK")
    print("-" * 70)

    for name, dt_values in periods.items():

        subset = df[df["dt"].isin(dt_values)]

        unique_values = subset["Pairflow"].unique()

        print(
            f"{name}: "
            f"{len(unique_values)} unique Pairflow values"
        )

        if len(unique_values) <= 10:
            print("Values:", unique_values)

    # ---------------------------------------------------------
    # 9. Relationship between Pairflow and label
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PAIRFLOW vs LABEL")
    print("-" * 70)

    pairflow_label = (
        df.groupby(["Pairflow", "label"])
        .size()
        .reset_index(name="count")
    )

    print(
        pairflow_label
        .sort_values("count", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 10. Save report
    # ---------------------------------------------------------
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "pairflow_investigation.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("PAIRFLOW INVESTIGATION - DATASET 1\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Raw shape: {df.shape}\n")
        f.write(f"Duplicates removed: {before - len(df)}\n\n")

        f.write("Overall statistics:\n")
        f.write(df["Pairflow"].describe().to_string())
        f.write("\n\n")

        f.write("Statistics by label:\n")
        f.write(label_stats.to_string())
        f.write("\n\n")

        for name, dt_values in periods.items():

            subset = df[df["dt"].isin(dt_values)]

            f.write(f"\n{name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Rows: {len(subset)}\n")
            f.write(f"dt range: {subset['dt'].min()} -> {subset['dt'].max()}\n")
            f.write(f"Mean: {subset['Pairflow'].mean()}\n")
            f.write(f"Std: {subset['Pairflow'].std()}\n")
            f.write(f"Min: {subset['Pairflow'].min()}\n")
            f.write(f"Max: {subset['Pairflow'].max()}\n")
            f.write(f"Unique values: {subset['Pairflow'].nunique()}\n")

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)

    print(f"\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    main()