
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "results" / "models" / "decision_tree"
METRICS_DIR = ROOT / "results" / "metrics" / "decision_tree"
REPORT_DIR = ROOT / "results" / "reports" / "decision_tree"

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


def false_positive_rates(cm):
    """Calculate false-positive rates for each class."""
    rates = []

    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denominator = fp + tn
        rates.append(float(fp / denominator) if denominator else 0.0)

    return rates


def train_dataset(name, config):
    print(f"\n{'=' * 60}")
    print(f"DECISION TREE — {name.upper()}")
    print(f"{'=' * 60}")

    dataset_dir = DATA_DIR / name

    train = pd.read_csv(dataset_dir / "train.csv")
    validation = pd.read_csv(dataset_dir / "validation.csv")
    test = pd.read_csv(dataset_dir / "test.csv")

    features = config["features"]

    X_train = train[features]
    y_train = train["label"]

    X_val = validation[features]
    y_val = validation["label"]

    X_test = test[features]
    y_test = test["label"]

    for split_name, X, y in [
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        if X.isna().any().any() or not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name}: {split_name} features contain missing "
                "or non-finite values."
            )

        if y.isna().any():
            raise ValueError(f"{name}: {split_name} labels contain missing values.")

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Input features:     {len(features)}")

    model = DecisionTreeClassifier(random_state=42)

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

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fpr_values = false_positive_rates(cm)

    metrics = {
        "dataset": name,
        "model": "Decision Tree",
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
            str(label): float(rate)
            for label, rate in zip(labels, fpr_values)
        },
        "training_time_seconds": float(training_time),
        "test_inference_time_seconds": float(inference_time),
        "class_names": [str(label) for label in labels],
        "confusion_matrix": cm.tolist(),
        "tree_depth": int(model.get_depth()),
        "number_of_leaves": int(model.get_n_leaves()),
    }

    # Additional binary-specific metrics for Dataset 1.
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
        metrics["false_positive_rate"] = fpr_values[
            labels.index(1)
        ] if 1 in labels else 0.0

    model_path = MODEL_DIR / f"decision_tree_{name}.joblib"
    metrics_path = METRICS_DIR / f"decision_tree_{name}_metrics.json"
    report_path = REPORT_DIR / f"decision_tree_{name}_confusion_matrix.csv"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(
        cm,
        index=[f"Actual_{label}" for label in labels],
        columns=[f"Predicted_{label}" for label in labels],
    ).to_csv(report_path)

    print(f"\nTest accuracy:     {accuracy:.4%}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4%}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall:    {recall:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}")
    print(f"Training time:      {training_time:.2f} seconds")
    print(f"Test inference time: {inference_time:.2f} seconds")
    print(f"Tree depth:         {model.get_depth()}")
    print(f"Number of leaves:   {model.get_n_leaves()}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nSaved model:", model_path.relative_to(ROOT))
    print("Saved metrics:", metrics_path.relative_to(ROOT))
    print("Saved confusion matrix:", report_path.relative_to(ROOT))


def main():
    for name, config in DATASETS.items():
        train_dataset(name, config)

    print("\nDecision Tree training completed for both datasets.")


if __name__ == "__main__":
    main()
