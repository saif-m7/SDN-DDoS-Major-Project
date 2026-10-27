
import os
import json
import time
import random
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

SEED = 42
EPOCHS_LIST = [10, 20, 30, 40, 50]
BATCH_SIZE = 256
LEARNING_RATE = 0.001

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

LABEL_COLUMN = "label"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(path, split_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} file not found:\n{path}"
        )

    print(f"Loading {split_name}: {path}")

    df = pd.read_csv(path)

    required_columns = FEATURES + [LABEL_COLUMN]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing required columns: "
            f"{missing_columns}"
        )

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name} contains NaN or infinite feature values."
        )

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"{split_name} contains unexpected labels: "
            f"{np.unique(y)}"
        )

    print(f"  Rows: {len(df):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) > 0 else 0.0
    )

    return {
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1_score": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    seed_everything()

    # Project root:
    # SDN-DDOS-Major-Project
    project_root = Path(__file__).resolve().parents[2]

    # MinMaxScaler-preprocessed Dataset 1
    data_dir = (
        project_root
        / "data"
        / "processed"
        / "experiments"
        / "dataset1_minmax"
    )

    # Separate output directory.
    # This does not overwrite the StandardScaler baseline.
    output_dir = (
        project_root
        / "results"
        / "experiments"
        / "dataset1_minmax_dnn"
    )

    models_dir = output_dir / "models"
    histories_dir = output_dir / "histories"

    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DNN EPOCH EXPERIMENT - MINMAX SCALER")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epoch experiments: {EPOCHS_LIST}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)

    # Load the same train/validation/test split files
    X_train, y_train = load_split(
        data_dir / "train.csv",
        "Training",
    )

    X_val, y_val = load_split(
        data_dir / "validation.csv",
        "Validation",
    )

    X_test, y_test = load_split(
        data_dir / "test.csv",
        "Test",
    )

    if X_train.shape[1] != len(FEATURES):
        raise ValueError(
            f"Expected {len(FEATURES)} features, "
            f"but received {X_train.shape[1]}."
        )

    results = []

    # --------------------------------------------------------
    # Train a fresh DNN for each epoch count
    # --------------------------------------------------------

    for epoch_count in EPOCHS_LIST:
        print("\n" + "=" * 70)
        print(f"TRAINING MINMAX DNN: {epoch_count} EPOCHS")
        print("=" * 70)

        tf.keras.backend.clear_session()
        seed_everything()

        model = build_dnn(
            input_dim=len(FEATURES),
            num_classes=2,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        start_time = time.time()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epoch_count,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=1,
        )

        training_time = time.time() - start_time

        # Evaluate on the test set
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # Save model separately for each epoch count
        model_path = (
            models_dir
            / f"dnn_dataset1_minmax_epochs_{epoch_count}.keras"
        )

        model.save(model_path)

        # Save training history
        history_data = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }

        history_path = (
            histories_dir
            / f"dnn_dataset1_minmax_epochs_{epoch_count}_history.json"
        )

        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history_data, file, indent=4)

        # Store metrics
        result = {
            "preprocessing": "MinMaxScaler",
            "epochs": epoch_count,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "training_time_seconds": float(training_time),
            "final_train_accuracy": float(
                history.history["accuracy"][-1]
            ),
            "final_validation_accuracy": float(
                history.history["val_accuracy"][-1]
            ),
            **metrics,
        }

        results.append(result)

        print("\nExperiment results:")
        print(f"Epochs: {epoch_count}")
        print(
            f"Training accuracy: "
            f"{result['final_train_accuracy']:.6f}"
        )
        print(
            f"Validation accuracy: "
            f"{result['final_validation_accuracy']:.6f}"
        )
        print(
            f"Test accuracy: "
            f"{metrics['test_accuracy']:.6f}"
        )
        print(
            f"Precision: {metrics['test_precision']:.6f}"
        )
        print(
            f"Recall: {metrics['test_recall']:.6f}"
        )
        print(
            f"F1-score: {metrics['test_f1_score']:.6f}"
        )
        print(
            f"False positive rate: "
            f"{metrics['false_positive_rate']:.6f}"
        )
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved: {model_path}")

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    csv_path = output_dir / "d1_minmax_epoch_comparison.csv"
    json_path = output_dir / "d1_minmax_epoch_comparison.json"
    xlsx_path = output_dir / "d1_minmax_epoch_comparison.xlsx"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    results_df.to_excel(xlsx_path, index=False)

    # --------------------------------------------------------
    # Plot accuracy against epoch count
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["epochs"],
        results_df["final_train_accuracy"],
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        results_df["epochs"],
        results_df["final_validation_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )

    plt.plot(
        results_df["epochs"],
        results_df["test_accuracy"],
        marker="^",
        label="Test Accuracy",
    )

    plt.title("MinMaxScaler DNN: Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(EPOCHS_LIST)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "d1_minmax_accuracy_vs_epochs.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MINMAX DNN EPOCH EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        results_df[
            [
                "epochs",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1_score",
                "false_positive_rate",
                "training_time_seconds",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {json_path}")
    print(f"Excel: {xlsx_path}")
    print(f"Plot:  {plot_path}")
    print(f"Models: {models_dir}")
    print(f"Histories: {histories_dir}")


if __name__ == "__main__":
    main()
