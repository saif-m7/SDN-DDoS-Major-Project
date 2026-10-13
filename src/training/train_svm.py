
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "results" / "models" / "svm"
METRICS_DIR = ROOT / "results" / "metrics" / "svm"
REPORT_DIR = ROOT / "results" / "reports" / "svm"

for directory in [MODEL_DIR, METRICS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "dataset1": {
        "features": [
            "switch", "pktcount", "bytecount", "dur", "dur_nsec",
            "tot_dur", "flows", "packetins", "pktperflow",
            "byteperflow", "pktrate", "Pairflow", "Protocol",
            "port_no", "tx_bytes", "rx_bytes", "tx_kbps",
            "rx_kbps", "tot_kbps"
        ],
        "task": "Binary DDoS Detection",
    },
    "dataset2": {
        "features": [
            "total_length", "ttl", "proto", "csum", "src_port",
            "dst_port", "tcp_flag", "type_icmp", "code_icmp",
            "tx_bytes_ave"
        ],
        "task": "Six-Class Traffic Classification",
    },
}


def calculate_fpr_per_class(cm):
    rates = []

    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denominator = fp + tn
        rates.append(float(fp / denominator) if denominator else 0.0)

    return rates


def train_dataset(name, config):
    print(f"\n{'=' * 60}")
    print(f"LINEAR SVM — {name.upper()}")
    print(f"{'=' * 60}")

    dataset_dir = DATA_DIR / name

    train = pd.read_csv(dataset_dir / "train.csv")
    validation = pd.read_csv(dataset_dir / "validation.csv")
    test = pd.read_csv(dataset_dir / "test.csv")

    features = config["features"]

    X_train = train[features]
    X_val = validation[features]
    X_test = test[features]

    y_train = train["label"].astype(str)
    y_val = validation["label"].astype(str)
    y_test = test["label"].astype(str)

    for split_name, X in [
        ("training", X_train),
        ("validation", X_val),
        ("test", X_test),
    ]:
        if X.isna().any().any() or not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name}: {split_name} features contain missing "
                "or non-finite values."
            )

    encoder = None

    if name == "dataset1":
        # Preserve numeric binary labels.
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
    else:
        # Fit the encoder only on training labels.
        encoder = LabelEncoder()
        encoder.fit(y_train)

        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test)

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42,
        dual="auto",
    )

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Input features:     {len(features)}")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    validation_predictions = model.predict(X_val)

    start = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    labels = list(range(len(encoder.classes_))) if encoder else [0, 1]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fpr_values = calculate_fpr_per_class(cm)

    class_names = (
        encoder.classes_.tolist()
        if encoder is not None
        else ["0", "1"]
    )

    metrics = {
        "dataset": name,
        "model": "Linear SVM (LinearSVC)",
        "task": config["task"],
        "input_features": features,
        "num_input_features": len(features),
        "accuracy": float(accuracy),
        "validation_accuracy": float(
            accuracy_score(y_val, validation_predictions)
        ),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "false_positive_rate_macro": float(np.mean(fpr_values)),
        "false_positive_rate_per_class": {
            class_names[i]: float(fpr_values[i])
            for i in range(len(class_names))
        },
        "training_time_seconds": float(training_time),
        "test_inference_time_seconds": float(inference_time),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    if name == "dataset1":
        metrics["precision"] = float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_test, predictions, pos_label=1, zero_division=0)
        )
        metrics["false_positive_rate"] = fpr_values[1]

    model_path = MODEL_DIR / f"svm_{name}.joblib"
    metrics_path = METRICS_DIR / f"svm_{name}_metrics.json"
    report_path = REPORT_DIR / f"svm_{name}_confusion_matrix.csv"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Predicted_{c}" for c in class_names],
    ).to_csv(report_path)

    if encoder is not None:
        joblib.dump(encoder, MODEL_DIR / "dataset2_label_encoder.joblib")

    print(f"\nTest accuracy:        {accuracy:.4%}")
    print(f"Validation accuracy:  {metrics['validation_accuracy']:.4%}")
    print(f"Weighted precision:   {precision:.4f}")
    print(f"Weighted recall:      {recall:.4f}")
    print(f"Weighted F1-score:    {f1:.4f}")
    print(f"Training time:        {training_time:.2f} seconds")
    print(f"Test inference time:  {inference_time:.2f} seconds")

    if hasattr(model, "n_iter_"):
        print(f"Solver iterations:    {model.n_iter_}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nSaved model:", model_path.relative_to(ROOT))
    print("Saved metrics:", metrics_path.relative_to(ROOT))
    print("Saved confusion matrix:", report_path.relative_to(ROOT))


def main():
    for name, config in DATASETS.items():
        train_dataset(name, config)

    print("\nSVM training completed for both datasets.")


if __name__ == "__main__":
    main()
