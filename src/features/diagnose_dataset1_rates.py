import pandas as pd
import numpy as np

# ============================================================
# DATASET 1 - RATE FEATURE DIAGNOSIS
# ============================================================

DATASET_PATH = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\raw\dataset1\SDN-DDoS_Traffic_Dataset.csv"
)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET 1")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# 2. Check target distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())
print("\nPercentage:")
print(df["label"].value_counts(normalize=True).mul(100).round(2))

# ------------------------------------------------------------
# 3. Check negative values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 70)

for column in ["byte_per_flow", "pkt_rate"]:

    negative_count = (df[column] < 0).sum()
    positive_count = (df[column] > 0).sum()
    zero_count = (df[column] == 0).sum()

    print(f"\nFeature: {column}")
    print(f"Negative : {negative_count:,}")
    print(f"Positive : {positive_count:,}")
    print(f"Zero     : {zero_count:,}")

# ------------------------------------------------------------
# 4. Negative values by label
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE VALUES BY LABEL")
print("=" * 70)

for column in ["byte_per_flow", "pkt_rate"]:

    print(f"\n--- {column} ---")

    negative_by_label = (
        df[df[column] < 0]
        .groupby("label")
        .size()
    )

    total_by_label = df.groupby("label").size()

    result = pd.DataFrame({
        "negative_count": negative_by_label,
        "total_count": total_by_label
    })

    result["negative_percentage"] = (
        result["negative_count"] /
        result["total_count"] * 100
    ).round(2)

    print(result)

# ------------------------------------------------------------
# 5. Statistical summary by label
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICS BY LABEL")
print("=" * 70)

for column in ["byte_per_flow", "pkt_rate"]:

    print(f"\n--- {column} ---")

    stats = df.groupby("label")[column].agg(
        ["count", "min", "max", "mean", "median", "std"]
    )

    print(stats)

# ------------------------------------------------------------
# 6. Statistics by protocol
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICS BY PROTOCOL")
print("=" * 70)

for column in ["byte_per_flow", "pkt_rate"]:

    print(f"\n--- {column} ---")

    stats = df.groupby("Protocol")[column].agg(
        ["count", "min", "max", "mean", "median", "std"]
    )

    print(stats)

# ------------------------------------------------------------
# 7. Check relationship with possible formulas
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING POSSIBLE FEATURE RELATIONSHIPS")
print("=" * 70)

# Possible packet-rate calculation:
# packet_count / duration

df["calculated_pkt_rate"] = np.where(
    df["duration"] != 0,
    df["pkt_count"] / df["duration"],
    np.nan
)

# Possible byte-per-flow calculation:
# byte_count / flows

df["calculated_byte_per_flow"] = np.where(
    df["flows"] != 0,
    df["byte_count"] / df["flows"],
    np.nan
)

print("\nCorrelation with possible calculated packet rate:")

print(
    df[["pkt_rate", "calculated_pkt_rate"]]
    .corr()
)

print("\nCorrelation with possible calculated byte-per-flow:")

print(
    df[["byte_per_flow", "calculated_byte_per_flow"]]
    .corr()
)

# ------------------------------------------------------------
# 8. Compare actual vs calculated values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ACTUAL VS CALCULATED VALUES")
print("=" * 70)

print("\nSample comparison for packet rate:")

print(
    df[
        [
            "pkt_count",
            "duration",
            "pkt_rate",
            "calculated_pkt_rate"
        ]
    ].head(10)
)

print("\nSample comparison for byte per flow:")

print(
    df[
        [
            "byte_count",
            "flows",
            "byte_per_flow",
            "calculated_byte_per_flow"
        ]
    ].head(10)
)

# ------------------------------------------------------------
# 9. Inspect negative examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE BYTE_PER_FLOW EXAMPLES")
print("=" * 70)

negative_byte = df[df["byte_per_flow"] < 0]

columns_to_show = [
    "switch",
    "host",
    "pkt_count",
    "byte_count",
    "flows",
    "byte_per_flow",
    "pkt_rate",
    "duration",
    "Protocol",
    "label"
]

print(
    negative_byte[columns_to_show]
    .head(10)
)

print("\n" + "=" * 70)
print("NEGATIVE PKT_RATE EXAMPLES")
print("=" * 70)

negative_pkt = df[df["pkt_rate"] < 0]

print(
    negative_pkt[columns_to_show]
    .head(10)
)

# ------------------------------------------------------------
# 10. Check whether negative values occur together
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RELATIONSHIP BETWEEN NEGATIVE FEATURES")
print("=" * 70)

both_negative = (
    (df["byte_per_flow"] < 0) &
    (df["pkt_rate"] < 0)
).sum()

only_byte_negative = (
    (df["byte_per_flow"] < 0) &
    (df["pkt_rate"] >= 0)
).sum()

only_pkt_negative = (
    (df["byte_per_flow"] >= 0) &
    (df["pkt_rate"] < 0)
).sum()

neither_negative = (
    (df["byte_per_flow"] >= 0) &
    (df["pkt_rate"] >= 0)
).sum()

print(f"Both negative       : {both_negative:,}")
print(f"Only byte_per_flow  : {only_byte_negative:,}")
print(f"Only pkt_rate       : {only_pkt_negative:,}")
print(f"Neither negative    : {neither_negative:,}")

# ------------------------------------------------------------
# 11. Correlation with other important features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH RELATED FEATURES")
print("=" * 70)

related_features = [
    "pkt_count",
    "byte_count",
    "duration",
    "duration_nsec",
    "tot_duration",
    "flows",
    "packet_per_massg",
    "pktper_flow",
    "byte_per_flow",
    "pkt_rate",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
    "delay",
    "jitter",
    "packet_loss_rate",
    "label"
]

correlation = df[related_features].corr()

print(
    correlation[
        ["byte_per_flow", "pkt_rate"]
    ].sort_values(
        by="pkt_rate",
        ascending=False
    )
)

# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

print("""
No changes were made to the original CSV.

Next step:
Review the output to determine whether negative
byte_per_flow and pkt_rate values are valid dataset
characteristics, calculation artifacts, or problematic values.

Do NOT use abs(), clipping, deletion, or replacement yet.
""")