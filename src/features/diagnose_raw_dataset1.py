import pandas as pd
import numpy as np

PATH = "data/raw/dataset1/SDN-DDoS_Traffic_Dataset.csv"

print("=" * 80)
print("RAW DATASET 1 DIAGNOSTIC")
print("=" * 80)

df = pd.read_csv(PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

# ---------------------------------------------------------
# Data quality
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("1. DATA QUALITY")
print("=" * 80)

print("Missing values:", df.isna().sum().sum())

numeric = df.select_dtypes(include=np.number)

print("Infinite values:", np.isinf(numeric).sum().sum())

# ---------------------------------------------------------
# Feature statistics by label
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("2. RAW FEATURE STATISTICS BY LABEL")
print("=" * 80)

normal = df[df["label"] == 0]
ddos = df[df["label"] == 1]

features = [
    col for col in numeric.columns
    if col != "label"
]

results = []

for feature in features:

    normal_mean = normal[feature].mean()
    ddos_mean = ddos[feature].mean()

    normal_std = normal[feature].std()
    ddos_std = ddos[feature].std()

    correlation = df[feature].corr(df["label"])

    results.append({
        "feature": feature,
        "normal_mean": normal_mean,
        "ddos_mean": ddos_mean,
        "normal_std": normal_std,
        "ddos_std": ddos_std,
        "correlation": correlation,
        "abs_correlation": abs(correlation)
    })

result_df = pd.DataFrame(results)

print(
    result_df.sort_values(
        "abs_correlation",
        ascending=False
    ).to_string(index=False)
)

# ---------------------------------------------------------
# Check suspicious rate features
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("3. SUSPICIOUS RATE FEATURES")
print("=" * 80)

for feature in ["byte_per_flow", "pkt_rate"]:

    if feature in df.columns:

        negative_percentage = (
            (df[feature] < 0).mean() * 100
        )

        print(
            f"{feature}: "
            f"{negative_percentage:.2f}% negative"
        )

# ---------------------------------------------------------
# Categorical features
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("4. CATEGORICAL FEATURES")
print("=" * 80)

for feature in ["Protocol", "src_ip", "dst_ip"]:

    if feature in df.columns:

        print(f"\n{feature}:")
        print(df[feature].value_counts().head(10))

# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RAW DATASET DIAGNOSTIC COMPLETED")
print("=" * 80)