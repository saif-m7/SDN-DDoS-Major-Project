import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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

LABEL = "label"

ALL_FEATURES = [
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


# ============================================================
# FEATURE GROUPS
# ============================================================

EXPERIMENTS = {

    "All 10 Features": ALL_FEATURES,

    "Without TTL": [
        "total_length",
        "proto",
        "csum",
        "src_port",
        "dst_port",
        "tcp_flag",
        "type_icmp",
        "code_icmp",
        "tx_bytes_ave"
    ],

    "Without TTL, DST_PORT, CODE_ICMP": [
        "total_length",
        "proto",
        "csum",
        "src_port",
        "tcp_flag",
        "type_icmp",
        "tx_bytes_ave"
    ],

    "Traffic/Header Core Features": [
        "total_length",
        "csum",
        "src_port",
        "dst_port",
        "tcp_flag",
        "tx_bytes_ave"
    ],

    "Traffic Features Only": [
        "total_length",
        "csum",
        "tx_bytes_ave"
    ],

    "Protocol Only": [
        "proto"
    ]
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DATASET 2 FEATURE ABLATION ANALYSIS")
print("=" * 80)

print("\nLoading Dataset 2...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(
    f"Training rows : {len(train_df):,}"
)

print(
    f"Test rows     : {len(test_df):,}"
)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df[LABEL]
)

y_test = label_encoder.transform(
    test_df[LABEL]
)


print("\nClasses:")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{index} → {class_name}"
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, features in EXPERIMENTS.items():

    print("\n" + "=" * 80)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print("=" * 80)

    print(
        f"\nFeatures used ({len(features)}):"
    )

    for feature in features:
        print(
            f"  - {feature}"
        )

    X_train = train_df[
        features
    ]

    X_test = test_df[
        features
    ]

    # --------------------------------------------------------
    # Simple decision tree diagnostic
    # --------------------------------------------------------
    #
    # This is NOT one of your project models.
    #
    # It is only being used here as a diagnostic tool to
    # determine whether the selected features themselves
    # make the classes trivially separable.
    #
    # max_depth=10 prevents an unrestricted tree from simply
    # memorizing the training data.

    diagnostic_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )

    diagnostic_model.fit(
        X_train,
        y_train
    )

    predictions = diagnostic_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nDiagnostic accuracy: "
        f"{accuracy:.4f}"
    )

    results.append({
        "experiment": experiment_name,
        "num_features": len(features),
        "accuracy": float(accuracy)
    })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ABLATION SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "reports"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "dataset2_feature_ablation.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    f"\nResults saved to: "
    f"{RESULTS_PATH}"
)

print("\n" + "=" * 80)
print("DATASET 2 FEATURE ABLATION COMPLETED")
print("=" * 80)