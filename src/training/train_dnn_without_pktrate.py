

import os
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dnn.dnn_model import build_dnn



# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"

EPOCHS = 30
BATCH_SIZE = 256
REMOVED_FEATURE = "pktrate"

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DATASET 1 FEATURES
# Removed: pktrate
# Remaining: 18 features
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
    # "pktrate" removed for this experiment
    "Pairflow",
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    """Check whether all required columns exist."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_dataset1_without_pktrate():

    print("\n" + "=" * 70)
    print("DNN TRAINING - DATASET 1 WITHOUT PKTRATE")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    dataset_dir = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset1",
    )

    train_path = os.path.join(dataset_dir, "train.csv")
    val_path = os.path.join(dataset_dir, "validation.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    # --------------------------------------------------------
    # Separate output directories
    # --------------------------------------------------------

    MODEL_DIR = os.path.join(
        BASE_DIR,
        "results",
        "models",
        "feature_ablation",
    )

    METRICS_DIR = os.path.join(
        BASE_DIR,
        "results",
        "metrics",
        "feature_ablation",
    )

    HISTORY_DIR = os.path.join(
        BASE_DIR,
        "results",
        "reports",
        "feature_ablation",
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load existing Dataset 1 splits
    # --------------------------------------------------------

    print("\nLoading Dataset 1...")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Training rows   : {len(train_df):,}")
    print(f"Validation rows : {len(val_df):,}")
    print(f"Test rows       : {len(test_df):,}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = DATASET1_FEATURES + ["label"]

    validate_columns(
        train_df,
        required_columns,
        "Dataset 1 training data",
    )

    validate_columns(
        val_df,
        required_columns,
        "Dataset 1 validation data",
    )

    validate_columns(
        test_df,
        required_columns,
        "Dataset 1 test data",
    )

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train = train_df[
        DATASET1_FEATURES
    ].to_numpy(dtype=np.float32)

    y_train = train_df[
        "label"
    ].to_numpy(dtype=np.int32)

    X_val = val_df[
        DATASET1_FEATURES
    ].to_numpy(dtype=np.float32)

    y_val = val_df[
        "label"
    ].to_numpy(dtype=np.int32)

    X_test = test_df[
        DATASET1_FEATURES
    ].to_numpy(dtype=np.float32)

    y_test = test_df[
        "label"
    ].to_numpy(dtype=np.int32)

    print("\nInput shapes without pktrate:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print(
        f"\nNumber of input features: "
        f"{len(DATASET1_FEATURES)}"
    )

    # --------------------------------------------------------
    # Build DNN
    # Same architecture as original train_dnn.py
    # --------------------------------------------------------

    model = build_dnn(
        input_dim=len(DATASET1_FEATURES),
        num_classes=2,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nDNN architecture:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "dnn_dataset1_without_pktrate_best.keras",
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nStarting training without pktrate...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            model_checkpoint,
        ],
        verbose=1,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\nEvaluating on TEST data...")

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        false_positive_rate = fp / (fp + tn)
    else:
        false_positive_rate = 0.0

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 1",
        "task": "Binary DDoS Detection",
        "model": "DNN",
        "experiment": "Feature ablation",
        "removed_feature": REMOVED_FEATURE,
        "input_features": DATASET1_FEATURES,
        "num_input_features": len(DATASET1_FEATURES),
        "epochs_configured": EPOCHS,
        "epochs_completed": len(
            history.history.get("loss", [])
        ),
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "confusion_matrix": cm.tolist(),
    }

    # --------------------------------------------------------
    # Save metrics separately
    # --------------------------------------------------------

    metrics_path = os.path.join(
        METRICS_DIR,
        "dnn_dataset1_without_pktrate_metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------------
    # Save training history separately
    # --------------------------------------------------------

    history_path = os.path.join(
        HISTORY_DIR,
        "dnn_dataset1_without_pktrate_history.json",
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history.history, file, indent=4)

    # --------------------------------------------------------
    # Compare with original metrics if available
    # Original metrics file is read-only
    # --------------------------------------------------------

    baseline_path = os.path.join(
        BASE_DIR,
        "results",
        "metrics",
        "dnn_dataset1_metrics.json",
    )

    comparison_path = os.path.join(
        METRICS_DIR,
        "dnn_dataset1_baseline_vs_without_pktrate.json",
    )

    comparison = {
        "baseline_metrics_file": baseline_path,
        "feature_ablation_metrics_file": metrics_path,
        "baseline": None,
        "without_pktrate": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(
                false_positive_rate
            ),
            "num_input_features": len(
                DATASET1_FEATURES
            ),
        },
        "change_without_pktrate_minus_baseline": None,
    }

    if os.path.isfile(baseline_path):

        with open(
            baseline_path,
            "r",
            encoding="utf-8",
        ) as file:
            baseline = json.load(file)

        comparison["baseline"] = {
            key: baseline.get(key)
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
                "num_input_features",
            ]
        }

        comparison[
            "change_without_pktrate_minus_baseline"
        ] = {}

        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ]:
            old_value = baseline.get(key)
            new_value = metrics.get(key)

            if isinstance(old_value, (int, float)) and isinstance(
                new_value, (int, float)
            ):
                comparison[
                    "change_without_pktrate_minus_baseline"
                ][key] = float(new_value) - float(old_value)

    else:
        print(
            "\nWARNING: Original metrics JSON was not found. "
            "The comparison file will contain only the new results."
        )

    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, indent=4)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET 1 RESULTS - WITHOUT PKTRATE")
    print("=" * 70)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nNew files saved:")
    print(f"Model      : {checkpoint_path}")
    print(f"Metrics    : {metrics_path}")
    print(f"History    : {history_path}")
    print(f"Comparison : {comparison_path}")

    print("\nOriginal training script and artifacts were not modified.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_dataset1_without_pktrate()