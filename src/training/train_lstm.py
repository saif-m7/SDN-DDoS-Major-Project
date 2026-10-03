import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.lstm.lstm_model import build_lstm


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 10
INPUT_DIM = 18
NUM_CLASSES = 2

EPOCHS = 30
BATCH_SIZE = 256
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


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

MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "lstm"
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
# Load temporal sequences
# ============================================================

print("=" * 70)
print("LSTM TRAINING - DATASET 1")
print("=" * 70)

print("\nLoading temporal sequences...")


X_train = np.load(
    DATA_DIR / "X_train.npy"
)

y_train = np.load(
    DATA_DIR / "y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_validation.npy"
)

y_val = np.load(
    DATA_DIR / "y_validation.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)


# ============================================================
# Display shapes
# ============================================================

print("\nSequence shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:  ",
    X_val.shape
)

print(
    "y_val:  ",
    y_val.shape
)

print(
    "X_test: ",
    X_test.shape
)

print(
    "y_test: ",
    y_test.shape
)


# ============================================================
# Validate input dimensions
# ============================================================

expected_shape = (
    SEQUENCE_LENGTH,
    INPUT_DIM
)

actual_shape = X_train.shape[1:]


if actual_shape != expected_shape:
    raise ValueError(
        f"Unexpected training input shape.\n"
        f"Expected: {expected_shape}\n"
        f"Received: {actual_shape}"
    )


if X_val.shape[1:] != expected_shape:
    raise ValueError(
        f"Unexpected validation input shape.\n"
        f"Expected: {expected_shape}\n"
        f"Received: {X_val.shape[1:]}"
    )


if X_test.shape[1:] != expected_shape:
    raise ValueError(
        f"Unexpected test input shape.\n"
        f"Expected: {expected_shape}\n"
        f"Received: {X_test.shape[1:]}"
    )


# ============================================================
# Validate labels
# ============================================================

unique_train_labels = np.unique(y_train)
unique_val_labels = np.unique(y_val)
unique_test_labels = np.unique(y_test)

print("\nTraining labels:", unique_train_labels)
print("Validation labels:", unique_val_labels)
print("Test labels:", unique_test_labels)


expected_labels = np.array([0, 1])


if not np.array_equal(
    unique_train_labels,
    expected_labels
):
    raise ValueError(
        "Training labels must contain 0 and 1."
    )


if not np.all(
    np.isin(
        unique_val_labels,
        expected_labels
    )
):
    raise ValueError(
        "Validation labels contain unexpected values."
    )


if not np.all(
    np.isin(
        unique_test_labels,
        expected_labels
    )
):
    raise ValueError(
        "Test labels contain unexpected values."
    )


# ============================================================
# Build LSTM
# ============================================================

print("\nBuilding LSTM model...")

model = build_lstm(
    sequence_length=SEQUENCE_LENGTH,
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES
)


# ============================================================
# Compile model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel summary:")
model.summary()


# ============================================================
# Callbacks
# ============================================================

model_path = (
    MODEL_DIR
    / "lstm_dataset1_best.keras"
)

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


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("STARTING LSTM TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# ============================================================
# Test prediction
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING LSTM ON UNSEEN TEST PERIOD")
print("=" * 70)

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# Metrics
# ============================================================

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


# ============================================================
# False Positive Rate
# ============================================================

tn, fp, fn, tp = cm.ravel()

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)


# ============================================================
# Save metrics
# ============================================================

metrics = {
    "dataset": "Dataset 1",
    "model": "LSTM",

    "sequence_length": SEQUENCE_LENGTH,
    "input_features": INPUT_DIM,

    "excluded_features": [
        "Pairflow",
        "src",
        "dst"
    ],

    "evaluation_type": "time-aware",

    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(fpr),

    "confusion_matrix": cm.tolist()
}


metrics_path = (
    METRICS_DIR
    / "lstm_dataset1_metrics.json"
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


# ============================================================
# Save training history
# ============================================================

history_path = (
    REPORT_DIR
    / "lstm_dataset1_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )


# ============================================================
# Print results
# ============================================================

print("\n" + "-" * 70)
print("LSTM DATASET 1 RESULTS")
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


print("\nBest model saved to:")
print(model_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\nTraining history saved to:")
print(history_path)


print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETED")
print("=" * 70)