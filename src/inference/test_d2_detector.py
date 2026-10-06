import sys
import os
import pandas as pd

# Add project root to Python path
BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"

sys.path.insert(
    0,
    os.path.join(BASE_DIR, "sdn", "controller")
)

from d2_detector import D2Detector


# ============================================================
# PATH
# ============================================================

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dataset2",
    "TEST-DATA.csv"
)


# ============================================================
# D2 FEATURES
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("D2 REAL DATA DETECTOR VERIFICATION")
print("=" * 70)

print("\nLoading TEST-DATA.csv...")

df = pd.read_csv(TEST_DATA_PATH)

print(f"Test dataset shape: {df.shape}")


# ============================================================
# LOAD DETECTOR
# ============================================================

detector = D2Detector()


# ============================================================
# SELECT 12 RECORDS
# ============================================================

# Take 2 records from each class
sample_df = (
    df.groupby(TARGET_COLUMN, group_keys=False)
      .head(2)
      .reset_index(drop=True)
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("REAL DATA PREDICTIONS")
print("=" * 70)

correct = 0
total = len(sample_df)

for index, row in sample_df.iterrows():

    record = {
        feature: row[feature]
        for feature in FEATURE_COLUMNS
    }

    actual_label = row[TARGET_COLUMN]

    result = detector.predict(record)

    predicted_label = result["label"]
    confidence = result["confidence"]

    is_correct = (
        actual_label == predicted_label
    )

    if is_correct:
        correct += 1
        status = "✓ CORRECT"
    else:
        status = "✗ WRONG"

    print("\n" + "-" * 70)

    print(f"Record       : {index + 1}")
    print(f"Actual       : {actual_label}")
    print(f"Predicted    : {predicted_label}")
    print(f"Status       : {result['status']}")
    print(f"Confidence   : {confidence:.6f}")
    print(f"Result       : {status}")


# ============================================================
# FINAL RESULT
# ============================================================

accuracy = (
    correct / total
) * 100

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"Records tested : {total}")
print(f"Correct        : {correct}")
print(f"Incorrect      : {total - correct}")
print(f"Accuracy       : {accuracy:.2f}%")

print("=" * 70)