import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.cnn.cnn_model import build_cnn


# ============================================================
# Configuration
# ============================================================

EPOCHS = 30
BATCH_SIZE = 256
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "results" / "models" / "cnn"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset 1 Features
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


# ============================================================
# Dataset 2 Features
# ============================================================

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
# Utility Functions
# ============================================================

def validate_columns(df, features, label_column, dataset_name):
    """
    Verify that all required columns exist.
    """

    required_columns = features + [label_column]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: Missing columns: {missing_columns}"
        )


def prepare_cnn_input(X):
    """
    Convert tabular feature matrix:

        (samples, features)

    into CNN input:

        (samples, features, 1)
    """

    X = X.astype(np.float32)

    return X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )


# ============================================================
# Dataset 1 Training
# ============================================================

def train_dataset1():

    print("\n" + "=" * 70)
    print("CNN TRAINING - DATASET 1")
    print("=" * 70)

    train_path = PROCESSED_DIR / "dataset1" / "train.csv"
    val_path = PROCESSED_DIR / "dataset1" / "validation.csv"
    test_path = PROCESSED_DIR / "dataset1" / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    validate_columns(
        train_df,
        DATASET1_FEATURES,
        "label",
        "Dataset 1"
    )

    validate_columns(
        val_df,
        DATASET1_FEATURES,
        "label",
        "Dataset 1"
    )

    validate_columns(
        test_df,
        DATASET1_FEATURES,
        "label",
        "Dataset 1"
    )

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train = train_df[DATASET1_FEATURES].values
    X_val = val_df[DATASET1_FEATURES].values
    X_test = test_df[DATASET1_FEATURES].values

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # --------------------------------------------------------
    # CNN input shape
    # --------------------------------------------------------

    X_train = prepare_cnn_input(X_train)
    X_val = prepare_cnn_input(X_val)
    X_test = prepare_cnn_input(X_test)

    print("\nInput shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_cnn(
        input_dim=len(DATASET1_FEATURES),
        num_classes=2
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    model_path = MODEL_DIR / "cnn_dataset1_best.keras"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

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

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "dataset": "Dataset 1",
        "model": "1D-CNN",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "features": DATASET1_FEATURES
    }

    metrics_path = METRICS_DIR / "cnn_dataset1_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    history_path = REPORT_DIR / "cnn_dataset1_history.json"

    with open(history_path, "w") as file:
        json.dump(
            history.history,
            file,
            indent=4
        )

    print("\n" + "-" * 70)
    print("DATASET 1 CNN RESULTS")
    print("-" * 70)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"False Positive Rate:{fpr:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nBest model saved to:")
    print(model_path)

    print(f"\nMetrics saved to:")
    print(metrics_path)


# ============================================================
# Dataset 2 Training
# ============================================================

def train_dataset2():

    print("\n" + "=" * 70)
    print("CNN TRAINING - DATASET 2")
    print("=" * 70)

    train_path = PROCESSED_DIR / "dataset2" / "train.csv"
    val_path = PROCESSED_DIR / "dataset2" / "validation.csv"
    test_path = PROCESSED_DIR / "dataset2" / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    validate_columns(
        train_df,
        DATASET2_FEATURES,
        "label",
        "Dataset 2"
    )

    validate_columns(
        val_df,
        DATASET2_FEATURES,
        "label",
        "Dataset 2"
    )

    validate_columns(
        test_df,
        DATASET2_FEATURES,
        "label",
        "Dataset 2"
    )

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train = train_df[DATASET2_FEATURES].values
    X_val = val_df[DATASET2_FEATURES].values
    X_test = test_df[DATASET2_FEATURES].values

    # Dataset 2 labels are strings.
    # Convert them to integer class IDs.
    labels = sorted(train_df["label"].unique())

    label_mapping = {
        label: index
        for index, label in enumerate(labels)
    }

    y_train = train_df["label"].map(label_mapping).values
    y_val = val_df["label"].map(label_mapping).values
    y_test = test_df["label"].map(label_mapping).values

    # --------------------------------------------------------
    # CNN input shape
    # --------------------------------------------------------

    X_train = prepare_cnn_input(X_train)
    X_val = prepare_cnn_input(X_val)
    X_test = prepare_cnn_input(X_test)

    print("\nInput shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    print("\nLabel mapping:")
    print(label_mapping)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_cnn(
        input_dim=len(DATASET2_FEATURES),
        num_classes=6
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    model_path = MODEL_DIR / "cnn_dataset2_best.keras"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

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

    # Macro FPR for multiclass classification
    per_class_fpr = []

    for i in range(len(labels)):

        tp = cm[i, i]

        fn = np.sum(cm[i, :]) - tp

        fp = np.sum(cm[:, i]) - tp

        tn = np.sum(cm) - (
            tp + fn + fp
        )

        class_fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        per_class_fpr.append(
            float(class_fpr)
        )

    macro_fpr = float(
        np.mean(per_class_fpr)
    )

    metrics = {
        "dataset": "Dataset 2",
        "model": "1D-CNN",
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "macro_false_positive_rate": macro_fpr,
        "per_class_false_positive_rate": {
            labels[i]: per_class_fpr[i]
            for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "label_mapping": label_mapping,
        "features": DATASET2_FEATURES
    }

    metrics_path = METRICS_DIR / "cnn_dataset2_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    mapping_path = METRICS_DIR / "cnn_dataset2_label_mapping.json"

    with open(mapping_path, "w") as file:
        json.dump(
            label_mapping,
            file,
            indent=4
        )

    history_path = REPORT_DIR / "cnn_dataset2_history.json"

    with open(history_path, "w") as file:
        json.dump(
            history.history,
            file,
            indent=4
        )

    print("\n" + "-" * 70)
    print("DATASET 2 CNN RESULTS")
    print("-" * 70)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}")
    print(f"Macro FPR:          {macro_fpr:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nBest model saved to:")
    print(model_path)

    print(f"\nMetrics saved to:")
    print(metrics_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CNN TRAINING PIPELINE")
    print("=" * 70)

    train_dataset1()
    train_dataset2()

    print("\n" + "=" * 70)
    print("CNN TRAINING COMPLETED")
    print("=" * 70)