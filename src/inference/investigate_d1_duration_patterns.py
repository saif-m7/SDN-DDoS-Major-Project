import pandas as pd
import numpy as np

DATASET = "data/raw/dataset1/dataset_sdn.csv"

df = pd.read_csv(DATASET)

# Recalculate rounded duration
df["expected"] = (
    np.round(
        df["dur"] + df["dur_nsec"] / 1e9
    ) * 1e9
)

df["difference"] = (
    df["tot_dur"] - df["expected"]
) / 1e9

# 1. Check repeated duration combinations
print("\n1. REPEATED DURATION COMBINATIONS")
print("-" * 60)

group_cols = [
    "dt", "switch", "Protocol",
    "dur", "dur_nsec"
]

grouped = df.groupby(group_cols).agg(
    records=("tot_dur", "size"),
    unique_tot_dur=("tot_dur", "nunique"),
    unique_expected=("expected", "nunique"),
    unique_difference=("difference", "nunique")
).reset_index()

repeated = grouped[grouped["records"] > 1]

print("Repeated groups:", len(repeated))
print(
    "Groups with different tot_dur:",
    (repeated["unique_tot_dur"] > 1).sum()
)

print("\nFirst 20 repeated groups:")
print(repeated.head(20).to_string(index=False))


# 2. Check mismatch patterns by duration
print("\n2. MISMATCHES BY DUR")
print("-" * 60)

df["is_mismatch"] = ~np.isclose(
    df["tot_dur"],
    df["expected"],
    atol=1
)

duration_stats = df.groupby("dur").agg(
    records=("tot_dur", "size"),
    mismatches=("is_mismatch", "sum")
)

duration_stats["mismatch_percentage"] = (
    duration_stats["mismatches"]
    / duration_stats["records"] * 100
)

print(
    duration_stats.sort_values(
        "mismatch_percentage",
        ascending=False
    ).head(20).to_string()
)


# 3. Check whether mismatches are concentrated
# in specific time intervals
print("\n3. MISMATCHES BY DT")
print("-" * 60)

dt_stats = df.groupby("dt").agg(
    records=("tot_dur", "size"),
    mismatches=("is_mismatch", "sum")
)

dt_stats["mismatch_percentage"] = (
    dt_stats["mismatches"]
    / dt_stats["records"] * 100
)

print(
    dt_stats.sort_values(
        "mismatch_percentage",
        ascending=False
    ).head(20).to_string()
)


# 4. Inspect actual records with mismatches
print("\n4. SAMPLE MISMATCH RECORDS")
print("-" * 60)

columns = [
    "dt", "switch", "port_no", "Protocol",
    "dur", "dur_nsec", "tot_dur",
    "expected", "difference"
]

print(
    df[df["is_mismatch"]][columns]
    .head(40)
    .to_string(index=False)
)