from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset1_temporal"
)


# ============================================================
# Load sequences
# ============================================================

print("=" * 70)
print("LSTM TEMPORAL DATA ANALYSIS - DATASET 1")
print("=" * 70)


X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_validation.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


# ============================================================
# Basic information
# ============================================================

print("\nSequence shapes:")

print(f"Train:      X={X_train.shape}, y={y_train.shape}")
print(f"Validation: X={X_val.shape}, y={y_val.shape}")
print(f"Test:       X={X_test.shape}, y={y_test.shape}")


# ============================================================
# Class distribution
# ============================================================

def print_class_distribution(name, labels):

    normal = np.sum(labels == 0)
    malicious = np.sum(labels == 1)

    total = len(labels)

    print(f"\n{name}")
    print("-" * 50)

    print(f"Total sequences : {total}")
    print(
        f"Normal          : {normal} "
        f"({normal / total * 100:.2f}%)"
    )

    print(
        f"Malicious       : {malicious} "
        f"({malicious / total * 100:.2f}%)"
    )


print_class_distribution(
    "TRAIN",
    y_train
)

print_class_distribution(
    "VALIDATION",
    y_val
)

print_class_distribution(
    "TEST",
    y_test
)


# ============================================================
# Sequence label transitions
# ============================================================

def sequence_transition_analysis(name, labels):

    transitions = np.sum(
        labels[1:] != labels[:-1]
    )

    print(f"\n{name} label transitions")
    print("-" * 50)

    print(
        f"Transitions between consecutive "
        f"sequence labels: {transitions}"
    )


sequence_transition_analysis(
    "TRAIN",
    y_train
)

sequence_transition_analysis(
    "VALIDATION",
    y_val
)

sequence_transition_analysis(
    "TEST",
    y_test
)


# ============================================================
# Feature statistics
# ============================================================

feature_names = [
    "switch",
    "pktcount",
    "bytecount",
    "dur",
    "dur_nsec",
    "tot_dur",
    "flows",
    "packetins",
    "pktperflow",
    "byteperflow",
    "pktrate",
    "Pairflow",
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps"
]


# ============================================================
# Compare feature distributions
# ============================================================

print("\nFeature distribution comparison")
print("=" * 70)

print(
    f"{'Feature':<15}"
    f"{'Train Mean':>14}"
    f"{'Val Mean':>14}"
    f"{'Test Mean':>14}"
)

print("-" * 70)


# Average over both samples and time steps.
train_feature_mean = X_train.mean(axis=(0, 1))
val_feature_mean = X_val.mean(axis=(0, 1))
test_feature_mean = X_test.mean(axis=(0, 1))


for i, feature in enumerate(feature_names):

    print(
        f"{feature:<15}"
        f"{train_feature_mean[i]:>14.4f}"
        f"{val_feature_mean[i]:>14.4f}"
        f"{test_feature_mean[i]:>14.4f}"
    )


# ============================================================
# Standard deviation comparison
# ============================================================

print("\nFeature standard deviation comparison")
print("=" * 70)

print(
    f"{'Feature':<15}"
    f"{'Train Std':>14}"
    f"{'Val Std':>14}"
    f"{'Test Std':>14}"
)

print("-" * 70)


train_feature_std = X_train.std(axis=(0, 1))
val_feature_std = X_val.std(axis=(0, 1))
test_feature_std = X_test.std(axis=(0, 1))


for i, feature in enumerate(feature_names):

    print(
        f"{feature:<15}"
        f"{train_feature_std[i]:>14.4f}"
        f"{val_feature_std[i]:>14.4f}"
        f"{test_feature_std[i]:>14.4f}"
    )


# ============================================================
# Within-sequence temporal variation
# ============================================================

print("\nWithin-sequence temporal variation")
print("=" * 70)

# Difference between consecutive time steps.
train_temporal_diff = np.diff(
    X_train,
    axis=1
)

val_temporal_diff = np.diff(
    X_val,
    axis=1
)

test_temporal_diff = np.diff(
    X_test,
    axis=1
)


train_temporal_mean = np.mean(
    np.abs(train_temporal_diff),
    axis=(0, 1)
)

val_temporal_mean = np.mean(
    np.abs(val_temporal_diff),
    axis=(0, 1)
)

test_temporal_mean = np.mean(
    np.abs(test_temporal_diff),
    axis=(0, 1)
)


print(
    f"{'Feature':<15}"
    f"{'Train Δ':>14}"
    f"{'Val Δ':>14}"
    f"{'Test Δ':>14}"
)

print("-" * 70)


for i, feature in enumerate(feature_names):

    print(
        f"{feature:<15}"
        f"{train_temporal_mean[i]:>14.4f}"
        f"{val_temporal_mean[i]:>14.4f}"
        f"{test_temporal_mean[i]:>14.4f}"
    )


# ============================================================
# NaN / Infinite checks
# ============================================================

print("\nData integrity checks")
print("=" * 70)

print(
    "Train NaN:",
    np.isnan(X_train).sum()
)

print(
    "Validation NaN:",
    np.isnan(X_val).sum()
)

print(
    "Test NaN:",
    np.isnan(X_test).sum()
)

print(
    "Train Inf:",
    np.isinf(X_train).sum()
)

print(
    "Validation Inf:",
    np.isinf(X_val).sum()
)

print(
    "Test Inf:",
    np.isinf(X_test).sum()
)


print("\n" + "=" * 70)
print("TEMPORAL LSTM ANALYSIS COMPLETED")
print("=" * 70)