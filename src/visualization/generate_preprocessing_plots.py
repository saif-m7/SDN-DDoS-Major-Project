
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

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
    "tot_kbps"
]

LABEL_COLUMN = "label"

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "visualizations"
    / "preprocessing_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXPERIMENT PATHS
# ============================================================

EXPERIMENTS = {
    "StandardScaler": {
        "data": BASE_DIR / "data" / "processed" / "dataset1",
        "model": BASE_DIR / "results" / "experiments" /
                 "dataset1_epoch_comparison" / "models" /
                 "dnn_dataset1_epochs_50.keras"
    },

    "MinMaxScaler": {
        "data": BASE_DIR / "data" / "processed" / "experiments" /
                "dataset1_minmax",
        "model": BASE_DIR / "results" / "experiments" /
                 "dataset1_minmax_dnn" / "models" /
                 "dnn_dataset1_minmax_epochs_50.keras"
    },

    "RobustScaler": {
        "data": BASE_DIR / "data" / "processed" / "experiments" /
                "dataset1_robust",
        "model": BASE_DIR / "results" / "experiments" /
                 "dataset1_robust_dnn" / "models" /
                 "dnn_dataset1_robust_epochs_50.keras"
    },

    "Log Transformation": {
        "data": BASE_DIR / "data" / "processed" / "experiments" /
                "dataset1_log",
        "model": BASE_DIR / "results" / "experiments" /
                 "dataset1_log_dnn" / "models" /
                 "dnn_log_50_epochs.keras"
    }
}


# ============================================================
# 3. CORRELATION MATRIX
# ============================================================

def generate_correlation_matrix(name, train_df):

    correlation = train_df[FEATURES].corr(method="pearson")

    plt.figure(figsize=(17, 13))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        annot_kws={"size": 7}
    )

    plt.title(
        f"{name} - Feature Correlation Matrix",
        fontsize=16
    )

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / (
        name.lower().replace(" ", "_")
        + "_correlation_matrix.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation matrix: {path}")


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(name, y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "DDoS"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{name} - Confusion Matrix",
        fontsize=15
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    plt.tight_layout()

    path = OUTPUT_DIR / (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix: {path}")


# ============================================================
# 5. ROC CURVE
# ============================================================

def generate_roc_curve(name, y_true, probabilities):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    auc_value = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.4f})",
        linewidth=2
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{name} - ROC Curve",
        fontsize=15
    )

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / (
        name.lower().replace(" ", "_")
        + "_roc_curve.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve: {path}")
    print(f"{name} ROC-AUC: {auc_value:.5f}")


# ============================================================
# 6. RUN ALL FOUR EXPERIMENTS
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("DNN PREPROCESSING VISUALIZATION")
    print("=" * 65)

    for name, config in EXPERIMENTS.items():

        print(f"\nProcessing: {name}")

        data_dir = config["data"]
        model_path = config["model"]

        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"

        # Check required files
        for path in [train_path, test_path, model_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"\nRequired file not found:\n{path}\n"
                    "Check the experiment path configuration."
                )

        # Load processed datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Validate features and labels
        required_columns = FEATURES + [LABEL_COLUMN]

        for df, split_name in [
            (train_df, "Training"),
            (test_df, "Testing")
        ]:
            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"{name} {split_name} data missing: {missing}"
                )

        # Actual test labels
        y_test = test_df[LABEL_COLUMN].to_numpy(
            dtype=np.int32
        )

        # Test features
        X_test = test_df[FEATURES].to_numpy(
            dtype=np.float32
        )

        # Load saved DNN
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # Predict probabilities
        probabilities = model.predict(
            X_test,
            batch_size=256,
            verbose=1
        )

        # Positive class probability = DDoS (class 1)
        if probabilities.ndim == 2:
            y_probability = probabilities[:, 1]
        else:
            y_probability = probabilities.reshape(-1)

        # Binary predictions at threshold 0.5
        y_prediction = (
            y_probability >= 0.5
        ).astype(np.int32)

        # Generate three plots
        generate_correlation_matrix(
            name,
            train_df
        )

        generate_confusion_matrix(
            name,
            y_test,
            y_prediction
        )

        generate_roc_curve(
            name,
            y_test,
            y_probability
        )

    print("\n" + "=" * 65)
    print("ALL VISUALIZATIONS COMPLETED")
    print("=" * 65)

    print(f"\nOutput folder:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
