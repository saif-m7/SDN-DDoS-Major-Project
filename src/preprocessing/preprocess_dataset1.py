import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATASET 1 PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

RAW_DATA_PATH = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\raw\dataset1\SDN-DDoS_Traffic_Dataset.csv"
)

PROCESSED_DIR = (
    r"C:\PROJECTS\SDN-DDOS-Major-Project"
    r"\data\processed\dataset1"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Features selected for the project
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "switch",
    "host",
    "pkt_count",
    "byte_count",
    "duration",
    "duration_nsec",
    "tot_duration",
    "flows",
    "packet_per_massg",
    "pktper_flow",
    "pair_flow",
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
    "delay",
    "jitter",
    "packet_loss_rate"
]

TARGET_COLUMN = "label"


# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET 1")
print("=" * 70)

df = pd.read_csv(RAW_DATA_PATH)

print(f"Original shape: {df.shape}")


# ------------------------------------------------------------
# 4. Select required columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SELECTING FEATURES")
print("=" * 70)

df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

print(f"Shape after feature selection: {df.shape}")

print("\nSelected features:")
for feature in FEATURE_COLUMNS:
    print(f" - {feature}")


# ------------------------------------------------------------
# 5. Check missing and infinite values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

missing_values = df.isnull().sum().sum()

print(f"Total missing values: {missing_values}")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

infinite_values = (
    df[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print(f"Total infinite values: {infinite_values}")


# ------------------------------------------------------------
# 6. Encode Protocol
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENCODING PROTOCOL")
print("=" * 70)

print("Original Protocol values:")
print(df["Protocol"].value_counts())

# Explicit mapping
protocol_mapping = {
    "ICMP": 0,
    "TCP": 1,
    "UDP": 2
}

df["Protocol"] = df["Protocol"].map(protocol_mapping)

print("\nEncoded Protocol values:")
print(df["Protocol"].value_counts())


# ------------------------------------------------------------
# 7. Separate features and target
# ------------------------------------------------------------

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# ------------------------------------------------------------
# 8. Train / Validation / Test split
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING TRAIN / VALIDATION / TEST SPLITS")
print("=" * 70)

# First split:
# 70% Training
# 30% Temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second split:
# 15% Validation
# 15% Test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ------------------------------------------------------------
# 9. Check label distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION AFTER SPLITTING")
print("=" * 70)

print("\nTraining:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nValidation:")
print(y_val.value_counts(normalize=True).mul(100).round(2))

print("\nTest:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


# ------------------------------------------------------------
# 10. Feature scaling
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

# IMPORTANT:
# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaler to validation and test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fitted on training data only.")

print(f"\nScaled training shape   : {X_train_scaled.shape}")
print(f"Scaled validation shape : {X_val_scaled.shape}")
print(f"Scaled test shape       : {X_test_scaled.shape}")


# ------------------------------------------------------------
# 11. Convert scaled arrays back to DataFrames
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

# Add target column
X_train_scaled[TARGET_COLUMN] = y_train.reset_index(drop=True)
X_val_scaled[TARGET_COLUMN] = y_val.reset_index(drop=True)
X_test_scaled[TARGET_COLUMN] = y_test.reset_index(drop=True)


# ------------------------------------------------------------
# 12. Save processed datasets
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

train_path = os.path.join(
    PROCESSED_DIR,
    "train.csv"
)

val_path = os.path.join(
    PROCESSED_DIR,
    "validation.csv"
)

test_path = os.path.join(
    PROCESSED_DIR,
    "test.csv"
)

X_train_scaled.to_csv(train_path, index=False)
X_val_scaled.to_csv(val_path, index=False)
X_test_scaled.to_csv(test_path, index=False)

print(f"Training data   → {train_path}")
print(f"Validation data → {val_path}")
print(f"Test data       → {test_path}")


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
print("DATASET 1 PREPROCESSING COMPLETE")
print("=" * 70)