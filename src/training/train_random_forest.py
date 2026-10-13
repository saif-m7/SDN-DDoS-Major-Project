
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_ESTIMATORS = 200

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "results" / "models" / "random_forest"
METRICS_DIR = BASE_DIR / "results" / "metrics" / "random_forest"
REPORTS_DIR = BASE_DIR / "results" / "reports" / "random_forest"

for directory in (MODEL_DIR, METRICS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

DATASET1_FEATURES = [
    "switch", "pktcount", "bytecount", "dur", "dur_nsec",
    "tot_dur", "flows", "packetins", "pktperflow",
    "byteperflow", "pktrate", "Pairflow", "Protocol",
    "port_no", "tx_bytes", "rx_bytes", "tx_kbps",
    "rx_kbps", "tot_kbps",
]

DATASET2_FEATURES = [
    "total_length", "ttl", "proto", "csum", "src_port",
    "dst_port", "tcp_flag", "type_icmp", "code_icmp",
    "tx_bytes_ave",
]


# ============================================================
# LOAD EXISTING PROCESSED SPLITS
# ============================================================

def load_dataset(dataset_name, features):
    dataset_dir = DATA_DIR / dataset_name
    splits = {}

    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        df = pd.read_csv(path)

        required_columns = features + ["label"]
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {missing}"
            )

        X = df[features].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy()

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite feature values found in {path}"
            )

        splits[split] = (X, y)

        print(
            f"{split.capitalize():10}: "
            f"{len(df):,} rows, {len(features)} features"
        )

    return splits


# ============================================================
# TRAIN AND EVALUATE RANDOM FOREST
# ============================================================

def train_random_forest(dataset_name, features):
    print("\n" + "=" * 70)
    print(f"RANDOM FOREST - {dataset_name.upper()}")
    print("=" * 70)

    splits = load_dataset(dataset_name, features)

    X_train, y_train_raw = splits["train"]
    X_val, y_val_raw = splits["validation"]
    X_test, y_test_raw = splits["test"]

    # D1 uses existing binary labels.
    # D2 uses an encoder fitted only on training labels.
    encoder = None

    if dataset_name == "dataset1":
        for split_name, labels in [
            ("train", y_train_raw),
            ("validation", y_val_raw),
            ("test", y_test_raw),
        ]:
            if not set(np.unique(labels)).issubset({0, 1}):
                raise ValueError(
                    f"Unexpected D1 labels in {split_name}: "
                    f"{np.unique(labels)}"
                )

        y_train = y_train_raw.astype(np.int32)
        y_val = y_val_raw.astype(np.int32)
        y_test = y_test_raw.astype(np.int32)

        class_ids = [0, 1]
        class_names = ["0", "1"]

    else:
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train_raw)
        y_val = encoder.transform(y_val_raw)
        y_test = encoder.transform(y_test_raw)

        class_ids = list(range(len(encoder.classes_)))
        class_names = encoder.classes_.tolist()

        if len(class_ids) != 6:
            raise ValueError(
                f"Expected 6 D2 classes, found {class_names}"
            )

        mapping = {
            str(i): str(name)
            for i, name in enumerate(encoder.classes_)
        }

        mapping_path = (
            METRICS_DIR / "dataset2_label_mapping.json"
        )

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4)

        print("\nD2 label mapping:")
        print(mapping)

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=None,
    )

    print("\nTraining Random Forest...")
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    # Validation is used for a diagnostic check.
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    # Final evaluation on the existing test set.
    print("\nEvaluating on the test set...")
    start_time = time.perf_counter()

    y_pred = model.predict(X_test)

    inference_seconds = time.perf_counter() - start_time

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_ids,
    )

    accuracy = accuracy_score(y_test, y_pred)

    if len(class_ids) == 2:
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": fpr,
        }

    else:
        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        class_fprs = []

        for i in range(len(class_ids)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            class_fprs.append(
                float(fp / (fp + tn))
                if (fp + tn) else 0.0
            )

        metrics = {
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_score_weighted": float(f1),
            "false_positive_rate_macro": float(
                np.mean(class_fprs)
            ),
            "false_positive_rate_per_class": {
                str(class_names[i]): class_fprs[i]
                for i in range(len(class_ids))
            },
        }

    metrics.update({
        "dataset": dataset_name,
        "model": "Random Forest",
        "task": (
            "Binary DDoS Detection"
            if len(class_ids) == 2
            else "Six-Class Traffic Classification"
        ),
        "input_features": features,
        "num_input_features": len(features),
        "accuracy": float(accuracy),
        "validation_accuracy": float(val_accuracy),
        "training_time_seconds": float(training_seconds),
        "test_inference_time_seconds": float(inference_seconds),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    })

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR / f"random_forest_{dataset_name}.joblib"
    )
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = (
        METRICS_DIR / f"random_forest_{dataset_name}_metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix
    cm_path = (
        REPORTS_DIR /
        f"random_forest_{dataset_name}_confusion_matrix.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in class_names],
        columns=[f"predicted_{c}" for c in class_names],
    )
    cm_df.to_csv(cm_path)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"RANDOM FOREST RESULTS - {dataset_name.upper()}")
    print("=" * 70)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Training time      : {training_seconds:.2f} seconds")
    print(f"Test inference time: {inference_seconds:.2f} seconds")

    if len(class_ids) == 2:
        print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    else:
        print(
            "Macro false positive rate: "
            f"{metrics['false_positive_rate_macro']:.4f}"
        )

    print("\nConfusion matrix:")
    print(cm)

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Confusion matrix saved: {cm_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM FOREST TRAINING")
    print("D1 and D2 are processed independently.")
    print("=" * 70)

    train_random_forest(
        "dataset1",
        DATASET1_FEATURES,
    )

    train_random_forest(
        "dataset2",
        DATASET2_FEATURES,
    )

    print("\nRandom Forest training completed for D1 and D2.")
