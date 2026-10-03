import pandas as pd
import numpy as np

TRAIN_PATH = "data/processed/dataset1/train.csv"

print("=" * 80)
print("DATASET 1 LEARNING DIAGNOSTIC")
print("=" * 80)

df = pd.read_csv(TRAIN_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Separate features and label
X = df.drop(columns=["label"])
y = df["label"]

print("\nLabel distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# ---------------------------------------------------------
# 1. Check missing and infinite values
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("1. DATA QUALITY")
print("=" * 80)

print("Missing values:", X.isna().sum().sum())
print("Infinite values:", np.isinf(X.select_dtypes(include=np.number)).sum().sum())

# ---------------------------------------------------------
# 2. Feature statistics by class
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("2. FEATURE STATISTICS BY CLASS")
print("=" * 80)

normal = df[df["label"] == 0]
ddos = df[df["label"] == 1]

numeric_features = X.select_dtypes(include=np.number).columns

rows = []

for feature in numeric_features:

    normal_mean = normal[feature].mean()
    ddos_mean = ddos[feature].mean()

    normal_std = normal[feature].std()
    ddos_std = ddos[feature].std()

    difference = abs(ddos_mean - normal_mean)

    rows.append({
        "feature": feature,
        "normal_mean": normal_mean,
        "ddos_mean": ddos_mean,
        "normal_std": normal_std,
        "ddos_std": ddos_std,
        "mean_difference": difference
    })

stats_df = pd.DataFrame(rows)

print(
    stats_df.sort_values(
        "mean_difference",
        ascending=False
    ).to_string(index=False)
)

# ---------------------------------------------------------
# 3. Correlation with label
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("3. CORRELATION WITH LABEL")
print("=" * 80)

correlations = []

for feature in numeric_features:

    corr = df[feature].corr(df["label"])

    correlations.append({
        "feature": feature,
        "correlation": corr,
        "absolute_correlation": abs(corr)
    })

corr_df = pd.DataFrame(correlations)

print(
    corr_df.sort_values(
        "absolute_correlation",
        ascending=False
    ).to_string(index=False)
)

# ---------------------------------------------------------
# 4. Variance check
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("4. LOW VARIANCE FEATURES")
print("=" * 80)

variance_df = pd.DataFrame({
    "feature": numeric_features,
    "variance": [df[f].var() for f in numeric_features]
})

print(
    variance_df.sort_values(
        "variance"
    ).to_string(index=False)
)

# ---------------------------------------------------------
# 5. Protocol distribution by label
# ---------------------------------------------------------

if "Protocol" in df.columns:

    print("\n" + "=" * 80)
    print("5. PROTOCOL DISTRIBUTION BY LABEL")
    print("=" * 80)

    print(
        pd.crosstab(
            df["Protocol"],
            df["label"],
            normalize="index"
        )
    )

# ---------------------------------------------------------
# 6. Final conclusion
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETED")
print("=" * 80)