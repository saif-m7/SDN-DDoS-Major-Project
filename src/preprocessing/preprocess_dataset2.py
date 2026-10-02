import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATASET 2 PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

TRAIN_DATA_PATH = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\raw\dataset2\TRAIN-DATA.csv"
)

TEST_DATA_PATH = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\raw\dataset2\TEST-DATA.csv"
)

PROCESSED_DIR = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\processed\dataset2"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Selected features
# ------------------------------------------------------------

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "label"


# ------------------------------------------------------------
# 3. Load original train and test datasets
# ------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET 2")
print("=" * 70)

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Original TRAIN shape: {train_df.shape}")
print(f"Original TEST shape : {test_df.shape}")


# ------------------------------------------------------------
# 4. Select required columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SELECTING FEATURES")
print("=" * 70)

train_df = train_df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
test_df = test_df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

print(f"TRAIN shape after feature selection: {train_df.shape}")
print(f"TEST shape after feature selection : {test_df.shape}")

print("\nSelected features:")

for feature in FEATURE_COLUMNS:
    print(f" - {feature}")


# ------------------------------------------------------------
# 5. Check data quality
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()

print(f"TRAIN missing values: {train_missing}")
print(f"TEST missing values : {test_missing}")


# Check infinite values
numeric_columns = train_df.select_dtypes(
    include=["int64", "float64"]
).columns

train_infinite = (
    train_df[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

test_infinite = (
    test_df[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print(f"TRAIN infinite values: {train_infinite}")
print(f"TEST infinite values : {test_infinite}")


# ------------------------------------------------------------
# 6. Check labels
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nTRAIN labels:")
print(train_df[TARGET_COLUMN].value_counts())

print("\nTEST labels:")
print(test_df[TARGET_COLUMN].value_counts())


# ------------------------------------------------------------
# 7. Separate features and target
# ------------------------------------------------------------

X_train_full = train_df[FEATURE_COLUMNS].copy()
y_train_full = train_df[TARGET_COLUMN].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].copy()


# ------------------------------------------------------------
# 8. Create validation set from TRAIN only
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    stratify=y_train_full,
    random_state=42
)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ------------------------------------------------------------
# 9. Check label distribution after split
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION AFTER SPLITTING")
print("=" * 70)

print("\nTraining:")
print(
    y_train.value_counts(
        normalize=True
    ).mul(100).round(2)
)

print("\nValidation:")
print(
    y_val.value_counts(
        normalize=True
    ).mul(100).round(2)
)

print("\nTest:")
print(
    y_test.value_counts(
        normalize=True
    ).mul(100).round(2)
)


# ------------------------------------------------------------
# 10. Feature scaling
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaler to validation and test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fitted on TRAINING data only.")


# ------------------------------------------------------------
# 11. Convert scaled arrays to DataFrames
# ------------------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=FEATURE_COLUMNS
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=FEATURE_COLUMNS
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=FEATURE_COLUMNS
)


# Add labels
X_train_scaled[TARGET_COLUMN] = y_train.reset_index(drop=True)
X_val_scaled[TARGET_COLUMN] = y_val.reset_index(drop=True)
X_test_scaled[TARGET_COLUMN] = y_test.reset_index(drop=True)


# ------------------------------------------------------------
# 12. Save processed datasets
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

train_output = os.path.join(
    PROCESSED_DIR,
    "train.csv"
)

validation_output = os.path.join(
    PROCESSED_DIR,
    "validation.csv"
)

test_output = os.path.join(
    PROCESSED_DIR,
    "test.csv"
)

X_train_scaled.to_csv(
    train_output,
    index=False
)

X_val_scaled.to_csv(
    validation_output,
    index=False
)

X_test_scaled.to_csv(
    test_output,
    index=False
)

print(f"Training data   → {train_output}")
print(f"Validation data → {validation_output}")
print(f"Test data       → {test_output}")


# ------------------------------------------------------------
# 13. Final verification
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print(f"Training rows   : {len(X_train_scaled):,}")
print(f"Validation rows : {len(X_val_scaled):,}")
print(f"Test rows       : {len(X_test_scaled):,}")

print(f"\nNumber of input features: {len(FEATURE_COLUMNS)}")

print("\nTraining sample:")
print(X_train_scaled.head())

print("\nMissing values:")

print(
    "Train:",
    X_train_scaled.isnull().sum().sum()
)

print(
    "Validation:",
    X_val_scaled.isnull().sum().sum()
)

print(
    "Test:",
    X_test_scaled.isnull().sum().sum()
)

print("\n" + "=" * 70)
print("DATASET 2 PREPROCESSING COMPLETE")
print("=" * 70)