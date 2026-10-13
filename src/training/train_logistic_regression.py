
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "dataset1": [
        "switch", "pktcount", "bytecount", "dur", "dur_nsec",
        "tot_dur", "flows", "packetins", "pktperflow",
        "byteperflow", "pktrate", "Pairflow", "Protocol",
        "port_no", "tx_bytes", "rx_bytes", "tx_kbps",
        "rx_kbps", "tot_kbps",
    ],
    "dataset2": [
        "total_length", "ttl", "proto", "csum", "src_port",
        "dst_port", "tcp_flag", "type_icmp", "code_icmp",
        "tx_bytes_ave",
    ],
}

MODEL_DIR = ROOT / "results" / "models" / "logistic_regression"
METRICS_DIR = ROOT / "results" / "metrics" / "logistic_regression"
REPORT_DIR = ROOT / "results" / "reports" / "logistic_regression"

for directory in (MODEL_DIR, METRICS_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_split(dataset_name, split):
    path = ROOT / "data" / "processed" / dataset_name / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    return pd.read_csv(path)


def train_dataset(dataset_name, features):
    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION — {dataset_name.upper()}")
    print("=" * 60)

    train_df = load_split(dataset_name, "train")
    val_df = load_split(dataset_name, "validation")
    test_df = load_split(dataset_name, "test")

    X_train = train_df[features].to_numpy(dtype=np.float64)
    X_val = val_df[features].to_numpy(dtype=np.float64)
    X_test = test_df[features].to_numpy(dtype=np.float64)

    # Labels are kept separate from input features.
    y_train_raw = train_df["label"].astype(str).to_numpy()
    y_val_raw = val_df["label"].astype(str).to_numpy()
    y_test_raw = test_df["label"].astype(str).to_numpy()

    if not all(np.isfinite(X).all() for X in (X_train, X_val, X_test)):
        raise ValueError(f"{dataset_name} contains non-finite feature values.")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)

    # Encode validation and test labels using training classes only.
    try:
        y_val = encoder.transform(y_val_raw)
        y_test = encoder.transform(y_test_raw)
    except ValueError as exc:
        raise ValueError(
            f"Unexpected label in validation/test data for {dataset_name}."
        ) from exc

    # Scaling is essential for Logistic Regression.
    # Fit the scaler on training data only to avoid data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Input features:     {len(features)}")
    print("Training model...")

    start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    training_time = time.perf_counter() - start

    val_predictions = model.predict(X_val_scaled)

    start = time.perf_counter()
    test_predictions = model.predict(X_test_scaled)
    inference_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(
        y_test, test_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, average="weighted", zero_division=0
    )
    val_accuracy = accuracy_score(y_val, val_predictions)

    labels = np.arange(len(encoder.classes_))
    cm = confusion_matrix(y_test, test_predictions, labels=labels)

    # Per-class false-positive rate: FP / (FP + TN).
    false_positive_rates = {}
    for i, class_name in enumerate(encoder.classes_):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denominator = fp + tn
        false_positive_rates[str(class_name)] = (
            float(fp / denominator) if denominator else 0.0
        )

    metrics = {
        "algorithm": "Logistic Regression",
        "dataset": dataset_name,
        "test_accuracy": float(accuracy),
        "validation_accuracy": float(val_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "training_time_seconds": float(training_time),
        "test_inference_time_seconds": float(inference_time),
        "test_samples": int(len(X_test)),
        "input_features": features,
        "classes": encoder.classes_.tolist(),
        "false_positive_rate_per_class": false_positive_rates,
        "confusion_matrix": cm.tolist(),
        "iterations": model.n_iter_.tolist(),
    }

    print(f"\nTest accuracy:        {accuracy:.4%}")
    print(f"Validation accuracy:  {val_accuracy:.4%}")
    print(f"Weighted precision:   {precision:.4f}")
    print(f"Weighted recall:      {recall:.4f}")
    print(f"Weighted F1-score:    {f1:.4f}")
    print(f"Training time:        {training_time:.2f} seconds")
    print(f"Test inference time:  {inference_time:.2f} seconds")
    print(f"Solver iterations:    {model.n_iter_.tolist()}")
    print("\nConfusion matrix:")
    print(cm)

    # Save scaler and model together so future predictions use the same scaling.
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": encoder,
        "features": features,
    }

    model_path = MODEL_DIR / f"logistic_regression_{dataset_name}.joblib"
    metrics_path = METRICS_DIR / f"logistic_regression_{dataset_name}_metrics.json"
    cm_path = REPORT_DIR / f"logistic_regression_{dataset_name}_confusion_matrix.csv"

    joblib.dump(model_bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_,
    ).to_csv(cm_path)

    print(f"\nSaved model bundle: {model_path.relative_to(ROOT)}")
    print(f"Saved metrics:      {metrics_path.relative_to(ROOT)}")
    print(f"Saved confusion matrix: {cm_path.relative_to(ROOT)}")


def main():
    for dataset_name, features in DATASETS.items():
        train_dataset(dataset_name, features)

    print("\nLogistic Regression training completed for both datasets.")


if __name__ == "__main__":
    main()
