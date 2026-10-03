import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"

DATASET2_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dataset2"
)

TRAIN_PATH = os.path.join(
    DATASET2_DIR,
    "TRAIN-DATA.csv"
)

TEST_PATH = os.path.join(
    DATASET2_DIR,
    "TEST-DATA.csv"
)

FEATURES = [
    "total_length",
    "ttl",
    "proto",
    "csum",
    "src_port",
    "dst_port",
    "tcp_flag",
    "type_icmp",
    "code_icmp",
    "tx_bytes_ave"
]

LABEL_COLUMN = "label"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DATASET 2 DIAGNOSTIC ANALYSIS")
print("=" * 80)

print("\nLoading Dataset 2...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training rows : {len(train_df):,}")
print(f"Test rows     : {len(test_df):,}")

print(f"\nTraining shape: {train_df.shape}")
print(f"Test shape    : {test_df.shape}")


# ============================================================
# 1. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("1. LABEL DISTRIBUTION")
print("=" * 80)

print("\nTraining labels:")
print(train_df[LABEL_COLUMN].value_counts())

print("\nTest labels:")
print(test_df[LABEL_COLUMN].value_counts())


# ============================================================
# 2. DUPLICATES WITHIN EACH DATASET
# ============================================================

print("\n" + "=" * 80)
print("2. DUPLICATE ANALYSIS")
print("=" * 80)

train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(
    f"\nDuplicate rows in TRAIN: "
    f"{train_duplicates:,}"
)

print(
    f"Duplicate rows in TEST : "
    f"{test_duplicates:,}"
)


# ============================================================
# 3. CROSS TRAIN-TEST DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN-TEST OVERLAP ANALYSIS")
print("=" * 80)

print("\nChecking whether identical complete rows")
print("exist in both TRAIN and TEST...")

train_full = train_df.astype(str)
test_full = test_df.astype(str)

train_full_set = set(
    map(
        tuple,
        train_full.to_numpy()
    )
)

test_full_set = set(
    map(
        tuple,
        test_full.to_numpy()
    )
)

full_overlap = (
    train_full_set
    & test_full_set
)

print(
    f"\nIdentical complete rows "
    f"appearing in both TRAIN and TEST: "
    f"{len(full_overlap):,}"
)


# ============================================================
# 4. FEATURE-ONLY OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE-ONLY TRAIN-TEST OVERLAP")
print("=" * 80)

print(
    "\nChecking identical feature vectors "
    "between TRAIN and TEST..."
)

train_features = train_df[
    FEATURES
].astype(str)

test_features = test_df[
    FEATURES
].astype(str)

train_feature_set = set(
    map(
        tuple,
        train_features.to_numpy()
    )
)

test_feature_set = set(
    map(
        tuple,
        test_features.to_numpy()
    )
)

feature_overlap = (
    train_feature_set
    & test_feature_set
)

print(
    f"\nIdentical feature vectors "
    f"in both TRAIN and TEST: "
    f"{len(feature_overlap):,}"
)


# ============================================================
# 5. CONFLICTING LABEL CHECK
# ============================================================

print("\n" + "=" * 80)
print("5. CONFLICTING LABEL ANALYSIS")
print("=" * 80)

print(
    "\nChecking whether the same feature vector "
    "has different labels..."
)

combined = pd.concat(
    [
        train_df[FEATURES + [LABEL_COLUMN]],
        test_df[FEATURES + [LABEL_COLUMN]]
    ],
    ignore_index=True
)

label_counts = (
    combined
    .groupby(FEATURES)[LABEL_COLUMN]
    .nunique()
)

conflicting_vectors = (
    label_counts[label_counts > 1]
)

print(
    f"\nFeature vectors with conflicting labels: "
    f"{len(conflicting_vectors):,}"
)

if len(conflicting_vectors) == 0:
    print(
        "No conflicting labels found for identical "
        "feature vectors."
    )
else:
    print(
        "WARNING: Some identical feature vectors "
        "have different labels."
    )


# ============================================================
# 6. FEATURE UNIQUE VALUES
# ============================================================

print("\n" + "=" * 80)
print("6. FEATURE UNIQUENESS")
print("=" * 80)

for feature in FEATURES:

    unique_count = train_df[
        feature
    ].nunique()

    print(
        f"{feature:<20} "
        f"unique values: {unique_count:,}"
    )


# ============================================================
# 7. FEATURE CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("7. FEATURE VALUES BY CLASS")
print("=" * 80)

for feature in FEATURES:

    print("\n" + "-" * 80)
    print(f"FEATURE: {feature}")
    print("-" * 80)

    try:

        summary = (
            train_df
            .groupby(LABEL_COLUMN)[feature]
            .agg(
                [
                    "min",
                    "max",
                    "mean",
                    "median",
                    "std"
                ]
            )
        )

        print(summary)

    except Exception as error:

        print(
            f"Could not analyze {feature}: "
            f"{error}"
        )


# ============================================================
# 8. CATEGORICAL FEATURE CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("8. CATEGORICAL FEATURE / LABEL RELATIONSHIPS")
print("=" * 80)

categorical_features = [
    "proto",
    "tcp_flag",
    "type_icmp",
    "code_icmp"
]

for feature in categorical_features:

    print("\n" + "-" * 80)
    print(f"FEATURE: {feature}")
    print("-" * 80)

    distribution = pd.crosstab(
        train_df[feature],
        train_df[LABEL_COLUMN],
        normalize="index"
    ) * 100

    print(
        distribution.round(2)
    )


# ============================================================
# 9. NUMERIC CORRELATION WITH LABEL
# ============================================================

print("\n" + "=" * 80)
print("9. NUMERIC FEATURE CORRELATION")
print("=" * 80)

# Temporarily encode the six classes numerically
label_mapping = {
    "DDOS_ICMP": 0,
    "DDOS_TCP": 1,
    "DDOS_UDP": 2,
    "NORMAL_ICMP": 3,
    "NORMAL_TCP": 4,
    "NORMAL_UDP": 5
}

encoded_labels = train_df[
    LABEL_COLUMN
].map(label_mapping)

correlation_df = train_df[
    FEATURES
].copy()

correlation_df[
    "encoded_label"
] = encoded_labels

correlations = (
    correlation_df
    .corr(numeric_only=True)["encoded_label"]
    .drop("encoded_label")
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(
    "\nCorrelation with encoded class label:"
)

print(
    correlations
)


# ============================================================
# 10. CHECK FOR CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("10. CONSTANT FEATURE CHECK")
print("=" * 80)

constant_features = []

for feature in FEATURES:

    if train_df[feature].nunique() <= 1:
        constant_features.append(feature)

if constant_features:

    print(
        "\nConstant features found:"
    )

    for feature in constant_features:
        print(f" - {feature}")

else:

    print(
        "\nNo constant features found "
        "among the selected 10 features."
    )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(
    f"\nTRAIN rows                  : "
    f"{len(train_df):,}"
)

print(
    f"TEST rows                   : "
    f"{len(test_df):,}"
)

print(
    f"TRAIN duplicate rows        : "
    f"{train_duplicates:,}"
)

print(
    f"TEST duplicate rows         : "
    f"{test_duplicates:,}"
)

print(
    f"Complete row overlap        : "
    f"{len(full_overlap):,}"
)

print(
    f"Feature-vector overlap      : "
    f"{len(feature_overlap):,}"
)

print(
    f"Conflicting feature labels  : "
    f"{len(conflicting_vectors):,}"
)

print(
    f"Selected model features     : "
    f"{len(FEATURES)}"
)

print("\n" + "=" * 80)
print("DATASET 2 DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 80)