import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.dnn.dnn_model import build_dnn


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"

EPOCHS = 30
BATCH_SIZE = 256


# ============================================================
# DATASET 1 FEATURES
# ============================================================

# Dataset 1 contains:
#
# dt + 19 model features + label
#
# dt is intentionally NOT used by the DNN.
# It will be retained for future LSTM/GRU sequence construction.

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
# DATASET 2 FEATURES
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
# DIRECTORY SETUP
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "dnn"
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

HISTORY_DIR = os.path.join(
    BASE_DIR,
    "results",
    "reports"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    """
    Verify that all required columns exist in the dataset.
    """

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
# DATASET 1
# ============================================================

def train_dataset1():

    print("\n" + "=" * 70)
    print("TRAINING DNN - DATASET 1")
    print("=" * 70)

    dataset_dir = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset1"
    )

    train_path = os.path.join(
        dataset_dir,
        "train.csv"
    )

    val_path = os.path.join(
        dataset_dir,
        "validation.csv"
    )

    test_path = os.path.join(
        dataset_dir,
        "test.csv"
    )

    # --------------------------------------------------------
    # Load data
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
        "Dataset 1 training data"
    )

    validate_columns(
        val_df,
        required_columns,
        "Dataset 1 validation data"
    )

    validate_columns(
        test_df,
        required_columns,
        "Dataset 1 test data"
    )

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    # IMPORTANT:
    # dt is NOT included here.

    X_train = train_df[
        DATASET1_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    y_train = train_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    X_val = val_df[
        DATASET1_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    y_val = val_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    X_test = test_df[
        DATASET1_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    y_test = test_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    print("\nDataset 1 input shape:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print(
        f"\nNumber of DNN input features: "
        f"{len(DATASET1_FEATURES)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_dnn(
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

    print("\nDataset 1 DNN:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "dnn_dataset1_best.keras"
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting Dataset 1 training...")

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
            model_checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating Dataset 1 "
        "on TEST data..."
    )

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = 0.0

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 1",
        "task": "Binary DDoS Detection",
        "model": "DNN",
        "input_features": DATASET1_FEATURES,
        "num_input_features": len(DATASET1_FEATURES),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "confusion_matrix": cm.tolist()
    }

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET 1 FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score           : "
        f"{f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        METRICS_DIR,
        "dnn_dataset1_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        HISTORY_DIR,
        "dnn_dataset1_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"History saved to: "
        f"{history_path}"
    )

    print(
        f"Best model saved to: "
        f"{checkpoint_path}"
    )


# ============================================================
# DATASET 2
# ============================================================

def train_dataset2():

    print("\n" + "=" * 70)
    print("TRAINING DNN - DATASET 2")
    print("=" * 70)

    dataset_dir = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset2"
    )

    train_path = os.path.join(
        dataset_dir,
        "train.csv"
    )

    val_path = os.path.join(
        dataset_dir,
        "validation.csv"
    )

    test_path = os.path.join(
        dataset_dir,
        "test.csv"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading Dataset 2...")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Training rows   : {len(train_df):,}")
    print(f"Validation rows : {len(val_df):,}")
    print(f"Test rows       : {len(test_df):,}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = DATASET2_FEATURES + ["label"]

    validate_columns(
        train_df,
        required_columns,
        "Dataset 2 training data"
    )

    validate_columns(
        val_df,
        required_columns,
        "Dataset 2 validation data"
    )

    validate_columns(
        test_df,
        required_columns,
        "Dataset 2 test data"
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train_df["label"]
    )

    y_val = label_encoder.transform(
        val_df["label"]
    )

    y_test = label_encoder.transform(
        test_df["label"]
    )

    print("\nLabel encoding:")

    for index, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{index} → {class_name}"
        )

    # --------------------------------------------------------
    # Separate features
    # --------------------------------------------------------

    X_train = train_df[
        DATASET2_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    X_val = val_df[
        DATASET2_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    X_test = test_df[
        DATASET2_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    print("\nDataset 2 input shape:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print(
        f"\nNumber of DNN input features: "
        f"{len(DATASET2_FEATURES)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_dnn(
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

    print("\nDataset 2 DNN:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "dnn_dataset2_best.keras"
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting Dataset 2 training...")

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
            model_checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating Dataset 2 "
        "on TEST data..."
    )

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Multiclass False Positive Rate
    # --------------------------------------------------------

    # Calculate one-vs-rest FPR for each class.

    false_positive_rates = []

    for class_index in range(
        len(label_encoder.classes_)
    ):

        tp = cm[
            class_index,
            class_index
        ]

        fn = (
            np.sum(
                cm[class_index, :]
            ) - tp
        )

        fp = (
            np.sum(
                cm[:, class_index]
            ) - tp
        )

        tn = (
            np.sum(cm)
            - tp
            - fn
            - fp
        )

        if (fp + tn) > 0:
            class_fpr = (
                fp / (fp + tn)
            )
        else:
            class_fpr = 0.0

        false_positive_rates.append(
            float(class_fpr)
        )

    macro_fpr = float(
        np.mean(false_positive_rates)
    )

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "dataset": "Dataset 2",
        "task": "Six-Class Traffic and DDoS Classification",
        "model": "DNN",
        "input_features": DATASET2_FEATURES,
        "num_input_features": len(DATASET2_FEATURES),
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "false_positive_rate_macro": macro_fpr,
        "false_positive_rate_per_class": {
            class_name: false_positive_rates[index]
            for index, class_name
            in enumerate(
                label_encoder.classes_
            )
        },
        "classes": label_encoder.classes_.tolist(),
        "confusion_matrix": cm.tolist()
    }

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET 2 FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1-score  : "
        f"{f1:.4f}"
    )

    print(
        f"Macro FPR          : "
        f"{macro_fpr:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        METRICS_DIR,
        "dnn_dataset2_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_mapping = {
        str(index): class_name
        for index, class_name
        in enumerate(
            label_encoder.classes_
        )
    }

    mapping_path = os.path.join(
        METRICS_DIR,
        "dnn_dataset2_label_mapping.json"
    )

    with open(
        mapping_path,
        "w"
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        HISTORY_DIR,
        "dnn_dataset2_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"Label mapping saved to: "
        f"{mapping_path}"
    )

    print(
        f"History saved to: "
        f"{history_path}"
    )

    print(
        f"Best model saved to: "
        f"{checkpoint_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DNN TRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset 1
    # --------------------------------------------------------

    train_dataset1()

    # --------------------------------------------------------
    # Dataset 2
    # --------------------------------------------------------

    train_dataset2()

    print("\n" + "=" * 70)
    print("ALL DNN TRAINING COMPLETED")
    print("=" * 70)