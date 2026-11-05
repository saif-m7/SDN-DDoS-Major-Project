
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")

if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

required = [
    "dt", "switch", "Protocol", "port_no",
    "pktcount", "flows", "dur", "tot_dur",
    "pktperflow", "pktrate"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("=" * 70)
print("DATASET 1 - PKTPERFLOW INVESTIGATION")
print("=" * 70)

print(f"Total records: {len(df)}")

# 1. Overall statistics
print("\n1. OVERALL STATISTICS")
print("-" * 70)
print(df["pktperflow"].describe().to_string())

print("\nUnique values (first 30):")
print(
    df["pktperflow"].value_counts()
    .sort_index()
    .head(30)
    .to_string()
)

# 2. Negative, zero and positive values
print("\n2. VALUE CATEGORIES")
print("-" * 70)

print("Negative:", (df["pktperflow"] < 0).sum())
print("Zero:", (df["pktperflow"] == 0).sum())
print("Positive:", (df["pktperflow"] > 0).sum())

# 3. By protocol
print("\n3. BY PROTOCOL")
print("-" * 70)

print(
    df.groupby("Protocol")["pktperflow"]
    .agg(["count", "min", "max", "mean", "median", "std"])
    .round(3)
    .to_string()
)

# 4. By switch
print("\n4. BY SWITCH")
print("-" * 70)

print(
    df.groupby("switch")["pktperflow"]
    .agg(["count", "min", "max", "mean", "median"])
    .round(3)
    .to_string()
)

# 5. Relationship with packet count and flows
print("\n5. RELATIONSHIP WITH PKTCOUNT AND FLOWS")
print("-" * 70)

for feature in ["pktcount", "flows", "dur", "tot_dur"]:
    print(
        f"Correlation with {feature}: "
        f"{df['pktperflow'].corr(df[feature]):.6f}"
    )

# 6. Test candidate formulas
print("\n6. CANDIDATE FORMULA TESTS")
print("-" * 70)

candidates = {
    "pktcount / flows":
        df["pktcount"] / df["flows"].replace(0, np.nan),

    "pktcount / dur":
        df["pktcount"] / df["dur"].replace(0, np.nan),

    "pktcount / tot_dur_seconds":
        df["pktcount"] / (df["tot_dur"] / 1e9).replace(0, np.nan),

    "pktrate * 30":
        df["pktrate"] * 30,
}

for name, candidate in candidates.items():
    valid = candidate.notna() & df["pktperflow"].notna()
    exact = np.isclose(
        candidate[valid],
        df.loc[valid, "pktperflow"]
    ).sum()

    total = valid.sum()

    print(
        f"{name}: {exact}/{total} "
        f"({exact / total * 100:.2f}%)"
    )

# 7. Group consistency
print("\n7. GROUP CONSISTENCY")
print("-" * 70)

for group_columns in [
    ["dt", "switch", "Protocol"],
    ["dt", "switch", "Protocol", "port_no"],
]:
    grouped = df.groupby(group_columns)["pktperflow"].agg(
        records="size",
        unique_values="nunique",
        minimum="min",
        maximum="max"
    )

    inconsistent = grouped[grouped["unique_values"] > 1]

    print(f"\nGrouping by: {group_columns}")
    print("Total groups:", len(grouped))
    print("Inconsistent groups:", len(inconsistent))

    if not inconsistent.empty:
        print(inconsistent.head(15).to_string())

# 8. Negative-value samples
print("\n8. NEGATIVE VALUE SAMPLES")
print("-" * 70)

negative = df[df["pktperflow"] < 0]

if negative.empty:
    print("No negative values.")
else:
    print(
        negative[
            ["dt", "switch", "Protocol", "pktcount",
             "flows", "dur", "pktperflow", "pktrate"]
        ].head(25).to_string(index=False)
    )

# 9. Final summary
print("\n9. SUMMARY")
print("-" * 70)

print("Negative records:", len(negative))
print("Zero records:", (df["pktperflow"] == 0).sum())
print("Positive records:", (df["pktperflow"] > 0).sum())
print("Unique timestamps:", df["dt"].nunique())
print("Unique switches:", df["switch"].nunique())

print("\nInvestigation completed.")
