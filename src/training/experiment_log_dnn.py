
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.dnn.dnn_model import build_dnn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiments"
    / "dataset1_log"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "dataset1_log_dnn"
)

MODEL_DIR = RESULTS_DIR / "models"
HISTORY_DIR = RESULTS_DIR / "histories"

FEATURES = [
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
    "tot_kbps",
]

TARGET = "label"

EPOCH_VALUES = [10, 20, 30, 40, 50]
BATCH_SIZE = 256
LEARNING_RATE = 0.001
RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Run experiment_log_preprocessing.py first."
        )

    df = pd.read_csv(path)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filename}: {missing_columns}"
        )

    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df[TARGET], errors="raise")

    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            f"{filename} contains NaN or infinite feature values."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"{filename} must have binary labels 0 and 1."
        )

    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int32)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 65)
    print("DATASET 1 - SHIFTED LOG TRANSFORMATION DNN EXPERIMENT")
    print("=" * 65)

    set_seed()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Load the already-preprocessed splits.
    X_train, y_train = load_dataset("train.csv")
    X_val, y_val = load_dataset("validation.csv")
    X_test, y_test = load_dataset("test.csv")

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")
    print(f"Features:        {len(FEATURES)}")
    print(f"Epoch settings:  {EPOCH_VALUES}")
    print()

    results = []

    for epoch_count in EPOCH_VALUES:
        print("-" * 65)
        print(f"Training DNN for {epoch_count} epochs")
        print("-" * 65)

        # Reset the seed so experiments start reproducibly.
        set_seed()

        model = build_dnn(
            input_dim=len(FEATURES),
            num_classes=2,
        )

        # Use the same optimizer configuration as the baseline.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        start_time = time.perf_counter()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epoch_count,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        training_time = time.perf_counter() - start_time

        # Predict the held-out test set.
        inference_start = time.perf_counter()
        probabilities = model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        inference_time = time.perf_counter() - inference_start

        # Handle either a two-class softmax or one-unit sigmoid output.
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            y_pred = (probabilities[:, 0] >= 0.5).astype(np.int32)
        elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
            y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
        else:
            raise ValueError(
                "Unexpected DNN output shape: "
                f"{probabilities.shape}. Check build_dnn()."
            )

        metrics = calculate_metrics(y_test, y_pred)

        train_accuracy = float(history.history["accuracy"][-1])
        validation_accuracy = float(
            history.history["val_accuracy"][-1]
        )

        # Save model and training history separately for each epoch setting.
        model_path = MODEL_DIR / f"dnn_log_{epoch_count}_epochs.keras"
        model.save(model_path)

        history_path = HISTORY_DIR / f"history_{epoch_count}_epochs.json"
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                file,
                indent=4,
            )

        result = {
            "preprocessing": "shifted_log1p",
            "epochs": epoch_count,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "false_positive_rate": metrics["false_positive_rate"],
            "true_negative": metrics["true_negative"],
            "false_positive": metrics["false_positive"],
            "false_negative": metrics["false_negative"],
            "true_positive": metrics["true_positive"],
            "training_time_seconds": round(training_time, 4),
            "test_inference_time_seconds": round(inference_time, 4),
        }

        results.append(result)

        print("\nResults:")
        for key, value in result.items():
            print(f"{key}: {value}")

        print(f"Saved model: {model_path}")

        # Release resources before the next experiment.
        del model
        tf.keras.backend.clear_session()

    # Save comparison table.
    results_df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "d1_log_epoch_comparison.csv"
    json_path = RESULTS_DIR / "d1_log_epoch_comparison.json"
    plot_path = RESULTS_DIR / "d1_log_accuracy_vs_epochs.png"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Excel is optional: CSV and JSON are saved first.
    try:
        xlsx_path = RESULTS_DIR / "d1_log_epoch_comparison.xlsx"
        results_df.to_excel(xlsx_path, index=False)
        print(f"Excel table saved: {xlsx_path}")
    except ImportError:
        print(
            "Excel export skipped because openpyxl is not installed. "
            "Install it with: python -m pip install openpyxl"
        )

    # Plot test and validation accuracy against epoch count.
    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["epochs"],
        results_df["validation_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )
    plt.plot(
        results_df["epochs"],
        results_df["test_accuracy"],
        marker="s",
        label="Test Accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("DNN Accuracy - Shifted Log Transformation")
    plt.xticks(EPOCH_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\n" + "=" * 65)
    print("LOG TRANSFORMATION DNN EXPERIMENT COMPLETED")
    print("=" * 65)
    print(f"CSV results:  {csv_path}")
    print(f"JSON results: {json_path}")
    print(f"Accuracy plot: {plot_path}")
    print("\nFinal comparison:")
    print(
        results_df[
            [
                "epochs",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
                "training_time_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
