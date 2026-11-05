
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Run this script from the project root directory."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "dt", "switch", "Protocol", "port_no", "flows"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("=" * 70)
print("DATASET 1 - FLOWS FEATURE INVESTIGATION")
print("=" * 70)

print(f"Total records: {len(df)}")


# --------------------------------------------------
# 1. Overall flows statistics
# --------------------------------------------------

print("\n1. OVERALL FLOWS STATISTICS")
print("-" * 70)

print(df["flows"].describe())

print("\nUnique flow values:")
print(df["flows"].value_counts().sort_index().to_string())


# --------------------------------------------------
# 2. Flows by protocol
# --------------------------------------------------

print("\n2. FLOWS BY PROTOCOL")
print("-" * 70)

print(
    df.groupby("Protocol")["flows"]
    .agg(["count", "min", "max", "mean", "std", "nunique"])
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 3. Flows by switch
# --------------------------------------------------

print("\n3. FLOWS BY SWITCH")
print("-" * 70)

print(
    df.groupby("switch")["flows"]
    .agg(["count", "min", "max", "mean", "nunique"])
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 4. Flows by timestamp, switch and protocol
# --------------------------------------------------

print("\n4. GROUP CONSISTENCY")
print("-" * 70)

group_columns = ["dt", "switch", "Protocol"]

grouped = (
    df.groupby(group_columns)["flows"]
    .agg(
        records="size",
        minimum="min",
        maximum="max",
        unique_values="nunique"
    )
    .reset_index()
)

print(f"Total groups: {len(grouped)}")

consistent = grouped[grouped["unique_values"] == 1]
inconsistent = grouped[grouped["unique_values"] > 1]

print(f"Groups with one flows value: {len(consistent)}")
print(f"Groups with multiple flows values: {len(inconsistent)}")

print("\nFirst 30 inconsistent groups:")

if inconsistent.empty:
    print("No inconsistent groups found.")
else:
    print(inconsistent.head(30).to_string(index=False))


# --------------------------------------------------
# 5. Inspect timestamp 2548
# --------------------------------------------------

print("\n5. SAMPLE RECORDS AT dt = 2548")
print("-" * 70)

sample = df[df["dt"] == 2548]

if sample.empty:
    print("No records found for dt = 2548.")
else:
    print(
        sample[
            ["dt", "switch", "Protocol", "port_no", "flows"]
        ].head(50).to_string(index=False)
    )


# --------------------------------------------------
# 6. Flow counts by switch and protocol
# --------------------------------------------------

print("\n6. FLOWS BY SWITCH AND PROTOCOL")
print("-" * 70)

print(
    df.groupby(["switch", "Protocol"])["flows"]
    .agg(["count", "min", "max", "mean", "nunique"])
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 7. Check whether flows changes over time
# --------------------------------------------------

print("\n7. FLOWS OVER TIME")
print("-" * 70)

time_summary = (
    df.groupby("dt")["flows"]
    .agg(
        records="size",
        minimum="min",
        maximum="max",
        unique_values="nunique"
    )
    .reset_index()
)

print("\nFirst 30 timestamps:")
print(time_summary.head(30).to_string(index=False))

print("\nLast 20 timestamps:")
print(time_summary.tail(20).to_string(index=False))


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n8. INVESTIGATION SUMMARY")
print("-" * 70)

print(f"Total records: {len(df)}")
print(f"Unique timestamps: {df['dt'].nunique()}")
print(f"Unique switches: {df['switch'].nunique()}")
print(f"Unique protocols: {df['Protocol'].nunique()}")
print(f"Minimum flows: {df['flows'].min()}")
print(f"Maximum flows: {df['flows'].max()}")
print(f"Consistent groups: {len(consistent)}")
print(f"Inconsistent groups: {len(inconsistent)}")

print("\nInvestigation completed.")
