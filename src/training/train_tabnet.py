import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from src.models.tabnet.tabnet_model import build_tabnet


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# Reduced for practical CPU training
MAX_EPOCHS = 30
BATCH_SIZE = 512
VIRTUAL_BATCH_SIZE = 128

PATIENCE = 7


# ============================================================
# Dataset feature definitions
# ============================================================

DATASET1_FEATURES = [
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


DATASET2_FEATURES = [
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
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset1"
)

DATASET2_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset2"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "tabnet"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Utility: Calculate FPR
# ============================================================

def calculate_binary_fpr(cm):
    """
    Calculate false-positive rate for binary classification.
    """

    if cm.shape != (2, 2):
        return None

    tn, fp, fn, tp = cm.ravel()

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return fp / denominator


# ============================================================
# Train Dataset 1
# ============================================================

def train_dataset1():

    print("\n" + "=" * 70)
    print("TABNET TRAINING - DATASET 1")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_path = DATASET1_DIR / "train.csv"
    val_path = DATASET1_DIR / "validation.csv"
    test_path = DATASET1_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\nDataset shapes:")
    print("Train      :", train_df.shape)
    print("Validation :", val_df.shape)
    print("Test       :", test_df.shape)

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in DATASET1_FEATURES
        if feature not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing Dataset 1 features: {missing_features}"
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X_train = train_df[
        DATASET1_FEATURES
    ].values.astype(np.float32)

    X_val = val_df[
        DATASET1_FEATURES
    ].values.astype(np.float32)

    X_test = test_df[
        DATASET1_FEATURES
    ].values.astype(np.float32)

    y_train = train_df["label"].values.astype(np.int64)
    y_val = val_df["label"].values.astype(np.int64)
    y_test = test_df["label"].values.astype(np.int64)

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if X_train.shape[1] != len(DATASET1_FEATURES):
        raise ValueError(
            f"Expected {len(DATASET1_FEATURES)} features, "
            f"received {X_train.shape[1]}"
        )

    print("\nNumber of features:", X_train.shape[1])

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding TabNet...")

    model = build_tabnet(
        num_classes=2,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting Dataset 1 training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        eval_name=[
            "train",
            "validation"
        ],
        eval_metric=[
            "accuracy"
        ],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        num_workers=0,
        drop_last=False
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nEvaluating Dataset 1...")

    y_pred = model.predict(
        X_test
    )

    y_pred = np.asarray(
        y_pred
    ).reshape(-1).astype(np.int64)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fpr = calculate_binary_fpr(
        cm
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "tabnet_dataset1"
    )

    model.save_model(
        str(model_path)
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 1",
        "model": "TabNet",
        "num_features": len(DATASET1_FEATURES),
        "features": DATASET1_FEATURES,
        "evaluation_type": "random stratified split",
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist()
    }

    metrics_path = (
        METRICS_DIR
        / "tabnet_dataset1_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TABNET DATASET 1 RESULTS")
    print("-" * 70)

    print(
        f"Accuracy:            {accuracy:.4f}"
    )

    print(
        f"Precision:           {precision:.4f}"
    )

    print(
        f"Recall:              {recall:.4f}"
    )

    print(
        f"F1-score:            {f1:.4f}"
    )

    print(
        f"False Positive Rate: {fpr:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nModel saved to:")
    print(model_path)

    print("\nMetrics saved to:")
    print(metrics_path)

    return model


# ============================================================
# Train Dataset 2
# ============================================================

def train_dataset2():

    print("\n" + "=" * 70)
    print("TABNET TRAINING - DATASET 2")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_path = DATASET2_DIR / "train.csv"
    val_path = DATASET2_DIR / "validation.csv"
    test_path = DATASET2_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\nDataset shapes:")
    print("Train      :", train_df.shape)
    print("Validation :", val_df.shape)
    print("Test       :", test_df.shape)

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in DATASET2_FEATURES
        if feature not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing Dataset 2 features: {missing_features}"
        )

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X_train = train_df[
        DATASET2_FEATURES
    ].values.astype(np.float32)

    X_val = val_df[
        DATASET2_FEATURES
    ].values.astype(np.float32)

    X_test = test_df[
        DATASET2_FEATURES
    ].values.astype(np.float32)

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train_df["label"]
    )

    y_val = label_encoder.transform(
        val_df["label"]
    )

    y_test = label_encoder.transform(
        test_df["label"]
    )

    num_classes = len(
        label_encoder.classes_
    )

    print("\nD2 label mapping:")

    for index, label in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{index} -> {label}"
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if X_train.shape[1] != len(DATASET2_FEATURES):
        raise ValueError(
            f"Expected {len(DATASET2_FEATURES)} features, "
            f"received {X_train.shape[1]}"
        )

    print(
        "\nNumber of features:",
        X_train.shape[1]
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding TabNet...")

    model = build_tabnet(
        num_classes=num_classes,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting Dataset 2 training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        eval_name=[
            "train",
            "validation"
        ],
        eval_metric=[
            "accuracy"
        ],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        num_workers=0,
        drop_last=False
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nEvaluating Dataset 2...")

    y_pred = model.predict(
        X_test
    )

    y_pred = np.asarray(
        y_pred
    ).reshape(-1).astype(np.int64)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Per-class FPR
    # --------------------------------------------------------

    per_class_fpr = {}

    for class_index, class_name in enumerate(
        label_encoder.classes_
    ):

        true_class = (
            y_test == class_index
        )

        predicted_class = (
            y_pred == class_index
        )

        fp = np.sum(
            (~true_class) & predicted_class
        )

        tn = np.sum(
            (~true_class) & (~predicted_class)
        )

        denominator = fp + tn

        class_fpr = (
            fp / denominator
            if denominator > 0
            else 0.0
        )

        per_class_fpr[
            str(class_name)
        ] = float(class_fpr)

    macro_fpr = float(
        np.mean(
            list(
                per_class_fpr.values()
            )
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "tabnet_dataset2"
    )

    model.save_model(
        str(model_path)
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 2",
        "model": "TabNet",
        "num_features": len(DATASET2_FEATURES),
        "features": DATASET2_FEATURES,
        "num_classes": num_classes,
        "evaluation_type": "supplied test split",
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "macro_false_positive_rate": macro_fpr,
        "per_class_false_positive_rate": per_class_fpr,
        "confusion_matrix": cm.tolist(),
        "label_mapping": {
            str(index): str(label)
            for index, label in enumerate(
                label_encoder.classes_
            )
        }
    }

    metrics_path = (
        METRICS_DIR
        / "tabnet_dataset2_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TABNET DATASET 2 RESULTS")
    print("-" * 70)

    print(
        f"Accuracy:              {accuracy:.4f}"
    )

    print(
        f"Weighted Precision:    {precision:.4f}"
    )

    print(
        f"Weighted Recall:       {recall:.4f}"
    )

    print(
        f"Weighted F1-score:     {f1:.4f}"
    )

    print(
        f"Macro FPR:             {macro_fpr:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nModel saved to:")
    print(model_path)

    print("\nMetrics saved to:")
    print(metrics_path)

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    mapping_path = (
        METRICS_DIR
        / "tabnet_dataset2_label_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                str(index): str(label)
                for index, label in enumerate(
                    label_encoder.classes_
                )
            },
            file,
            indent=4
        )

    print("\nLabel mapping saved to:")
    print(mapping_path)

    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train_dataset1()

    train_dataset2()

    print("\n" + "=" * 70)
    print("ALL TABNET TRAINING COMPLETED")
    print("=" * 70)