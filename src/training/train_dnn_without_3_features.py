
"""
DNN Feature Ablation Experiment - Dataset 1

Original DNN: 19 features
Experiment 1: 18 features (removed pktrate)
Experiment 2: 16 features (removed pktrate, byteperflow, Pairflow)

This script does not modify the original training script,
dataset files, or previously trained models.
"""

import os
import sys
import json
import random
from pathlib import Path

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

# Ensure the project root is available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dnn.dnn_model import build_dnn


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
BASE_DIR = str(PROJECT_ROOT)

EPOCHS = 30
BATCH_SIZE = 256

REMOVED_FEATURES = [
    "pktrate",
    "byteperflow",
    "Pairflow",
]


# ============================================================
# DATASET 1 FEATURES
# Original 19 features minus the 3 selected features = 16
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
    # "byteperflow" removed
    # "pktrate" removed
    # "Pairflow" removed
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
]


# ============================================================
# RANDOM SEEDS
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# HELPER FUNCTION
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    """Validate that all required columns are present."""

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
# TRAINING FUNCTION
# ============================================================

def train_dataset1_without_3_features():

    print("\n" + "=" * 72)
    print("DNN FEATURE ABLATION - DATASET 1")
    print("Removed features:", ", ".join(REMOVED_FEATURES))
    print("Remaining features:", len(DATASET1_FEATURES))
    print("=" * 72)

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

    model_dir = os.path.join(
        BASE_DIR,
        "results",
        "models",
        "feature_ablation",
    )

    metrics_dir = os.path.join(
        BASE_DIR,
        "results",
        "metrics",
        "feature_ablation",
    )

    history_dir = os.path.join(
        BASE_DIR,
        "results",
        "reports",
        "feature_ablation",
    )

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load the existing dataset splits
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
    # Prepare input features and labels
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

    print("\nInput shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    # --------------------------------------------------------
    # Build and compile the DNN
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

    print("\nDNN model summary:")
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
        model_dir,
        "dnn_dataset1_without_3_features_best.keras",
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

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

    accuracy = accuracy_score(y_test, predictions)

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 1",
        "task": "Binary DDoS Detection",
        "model": "DNN",
        "experiment": "Feature ablation",
        "removed_features": REMOVED_FEATURES,
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
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        metrics_dir,
        "dnn_dataset1_without_3_features_metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        history_dir,
        "dnn_dataset1_without_3_features_history.json",
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history.history, file, indent=4)

    # --------------------------------------------------------
    # Create comparison with the original DNN
    # --------------------------------------------------------

    baseline_path = os.path.join(
        BASE_DIR,
        "results",
        "metrics",
        "dnn_dataset1_metrics.json",
    )

    comparison_path = os.path.join(
        metrics_dir,
        "dnn_dataset1_baseline_vs_without_3_features.json",
    )

    comparison = {
        "baseline_metrics_file": baseline_path,
        "feature_ablation_metrics_file": metrics_path,
        "baseline": None,
        "without_3_features": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(
                false_positive_rate
            ),
            "num_input_features": len(DATASET1_FEATURES),
            "removed_features": REMOVED_FEATURES,
        },
        "change_without_3_features_minus_baseline": None,
    }

    if os.path.isfile(baseline_path):

        with open(
            baseline_path,
            "r",
            encoding="utf-8",
        ) as file:
            baseline = json.load(file)

        baseline_metrics = {
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

        comparison["baseline"] = baseline_metrics

        changes = {}

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
                changes[key] = float(new_value) - float(old_value)

        comparison[
            "change_without_3_features_minus_baseline"
        ] = changes

    else:
        print(
            "\nWARNING: Original baseline metrics file was not found."
        )

    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, indent=4)

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("DATASET 1 RESULTS - WITHOUT 3 FEATURES")
    print("=" * 72)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nRemoved features:")
    for feature in REMOVED_FEATURES:
        print(f"- {feature}")

    print("\nSaved artifacts:")
    print(f"Model      : {checkpoint_path}")
    print(f"Metrics    : {metrics_path}")
    print(f"History    : {history_path}")
    print(f"Comparison : {comparison_path}")

    print("\nOriginal script and previous experiment artifacts were not modified.")


# ============================================================
# RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    train_dataset1_without_3_features()
