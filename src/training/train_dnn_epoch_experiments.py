import os, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.models.dnn.dnn_model import build_dnn

SEED = 42
EPOCHS_LIST = [10, 20, 30, 40, 50]
BATCH_SIZE = 256
LEARNING_RATE = 0.001
FEATURES = [
    "switch", "pktcount", "bytecount", "dur", "dur_nsec", "tot_dur",
    "flows", "packetins", "pktperflow", "byteperflow", "pktrate",
    "Pairflow", "Protocol", "port_no", "tx_bytes", "rx_bytes",
    "tx_kbps", "rx_kbps", "tot_kbps",
]

def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

def load_split(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} file is missing columns: {missing}")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    if not np.isfinite(X).all():
        raise ValueError(f"{name} has NaN/inf values. Run existing preprocessing first.")
    return X, y

def main():
    root = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == "training" else Path.cwd()
    # This script should be placed in <project-root>/src/training/.
    data_dir = root / "data" / "processed" / "dataset1"
    out_dir = root / "results" / "experiments" / "dataset1_epoch_comparison"
    model_dir, history_dir = out_dir / "models", out_dir / "histories"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset 1 DNN epoch experiment")
    print("Project root:", root)
    X_train, y_train = load_split(data_dir / "train.csv", "Training")
    X_val, y_val = load_split(data_dir / "validation.csv", "Validation")
    X_test, y_test = load_split(data_dir / "test.csv", "Test")
    print(f"Rows: train={len(y_train):,}, validation={len(y_val):,}, test={len(y_test):,}")
    print("Early stopping is disabled. Existing production model will not be overwritten.")

    results = []
    for epoch_count in EPOCHS_LIST:
        print(f"\\n{'='*60}\\nTraining DNN for {epoch_count} epochs\\n{'='*60}")
        tf.keras.backend.clear_session()
        seed_everything()
        model = build_dnn(input_dim=len(FEATURES), num_classes=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        started = time.perf_counter()
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epoch_count, batch_size=BATCH_SIZE, shuffle=True, verbose=1
        )
        elapsed = time.perf_counter() - started

        probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        pred = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if fp + tn else 0.0

        model_path = model_dir / f"dnn_dataset1_epochs_{epoch_count}.keras"
        history_path = history_dir / f"dnn_dataset1_epochs_{epoch_count}_history.json"
        model.save(model_path)
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")

        results.append({
            "dataset": "Dataset 1", "model": "DNN",
            "epochs_requested": epoch_count,
            "epochs_completed": len(history.history["loss"]),
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "training_accuracy_last_epoch": float(history.history["accuracy"][-1]),
            "validation_accuracy_last_epoch": float(history.history["val_accuracy"][-1]),
            "training_loss_last_epoch": float(history.history["loss"][-1]),
            "validation_loss_last_epoch": float(history.history["val_loss"][-1]),
            "test_accuracy": float(acc), "test_precision": float(precision),
            "test_recall": float(recall), "test_f1_score": float(f1),
            "test_false_positive_rate": float(fpr),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
            "training_time_seconds": float(elapsed),
            "model_file": str(model_path.relative_to(root)),
            "history_file": str(history_path.relative_to(root)),
        })
        print(f"Test accuracy={acc:.4f}; precision={precision:.4f}; recall={recall:.4f}; F1={f1:.4f}; FPR={fpr:.4f}")
        print(f"Training time={elapsed:.2f}s; saved model={model_path}")
        del model

    df = pd.DataFrame(results)
    csv_path = out_dir / "d1_epoch_comparison.csv"
    json_path = out_dir / "d1_epoch_comparison.json"
    xlsx_path = out_dir / "d1_epoch_comparison.xlsx"
    plot_path = out_dir / "d1_accuracy_vs_epochs.png"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    try:
        df.to_excel(xlsx_path, index=False)
        print("Excel report:", xlsx_path)
    except Exception as exc:
        print("Excel export unavailable; CSV is saved. Reason:", exc)

    plt.figure(figsize=(9, 5))
    plt.plot(df["epochs_requested"], df["training_accuracy_last_epoch"], marker="o", label="Training accuracy")
    plt.plot(df["epochs_requested"], df["validation_accuracy_last_epoch"], marker="o", label="Validation accuracy")
    plt.plot(df["epochs_requested"], df["test_accuracy"], marker="o", label="Test accuracy")
    plt.xlabel("Epoch count"); plt.ylabel("Accuracy")
    plt.title("Dataset 1 DNN: Accuracy vs Epoch Count")
    plt.xticks(EPOCHS_LIST); plt.ylim(0, 1.02); plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout(); plt.savefig(plot_path, dpi=200); plt.close()

    print("\\nResults saved:")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Plot:", plot_path)
    print(df[["epochs_requested", "training_accuracy_last_epoch", "validation_accuracy_last_epoch", "test_accuracy", "training_time_seconds"]].to_string(index=False))

if __name__ == "__main__":
    main()
