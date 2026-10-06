import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dataset2",
    "test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "dnn",
    "dnn_dataset2_best.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "mitigation"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "d2_offline_mitigation.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "d2_offline_mitigation_summary.json"
)


# ============================================================
# D2 FEATURES
# ============================================================

D2_FEATURES = [
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

TARGET_COLUMN = "label"


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "DDOS_ICMP",
    "DDOS_TCP",
    "DDOS_UDP",
    "NORMAL_ICMP",
    "NORMAL_TCP",
    "NORMAL_UDP"
]


# ============================================================
# MITIGATION DECISION
# ============================================================

def get_mitigation_action(predicted_class):
    """
    Convert D2 attack classification into an
    offline mitigation decision.
    """

    if predicted_class == "DDOS_ICMP":
        return "BLOCK_ICMP"

    elif predicted_class == "DDOS_TCP":
        return "BLOCK_TCP"

    elif predicted_class == "DDOS_UDP":
        return "BLOCK_UDP"

    elif predicted_class == "NORMAL_ICMP":
        return "ALLOW_ICMP"

    elif predicted_class == "NORMAL_TCP":
        return "ALLOW_TCP"

    elif predicted_class == "NORMAL_UDP":
        return "ALLOW_UDP"

    return "UNKNOWN"


# ============================================================
# ATTACK STATUS
# ============================================================

def get_attack_status(predicted_class):

    if predicted_class.startswith("DDOS"):
        return "MALICIOUS"

    return "NORMAL"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("        D2 OFFLINE DDoS MITIGATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"D2 test dataset not found:\n{TEST_DATA_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"D2 trained model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\n[1/6] Loading D2 test dataset...")

    df = pd.read_csv(TEST_DATA_PATH)

    print(f"Test dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = D2_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    print("\n[2/6] Preparing D2 features...")

    X = df[D2_FEATURES].copy()
    y_true = df[TARGET_COLUMN].astype(str)

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isnull().sum().sum() > 0:
        print("Warning: Missing values detected.")
        X = X.fillna(X.median(numeric_only=True))

    X = X.astype(np.float32)

    # --------------------------------------------------------
    # Load trained D2 model
    # --------------------------------------------------------

    print("\n[3/6] Loading trained D2 DNN model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n[4/6] Performing D2 attack classification...")

    probabilities = model.predict(
        X,
        batch_size=1024,
        verbose=1
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_classes = [
        CLASS_NAMES[index]
        for index in predicted_indices
    ]

    confidence = np.max(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predicted_classes
    )

    print(f"\nClassification Accuracy: {accuracy:.4f}")

    # --------------------------------------------------------
    # Create mitigation results
    # --------------------------------------------------------

    print("\n[5/6] Generating offline mitigation decisions...")

    results = df.copy()

    results["predicted_class"] = predicted_classes

    results["confidence"] = confidence

    results["attack_status"] = [
        get_attack_status(predicted_class)
        for predicted_class in predicted_classes
    ]

    results["mitigation_action"] = [
        get_mitigation_action(predicted_class)
        for predicted_class in predicted_classes
    ]

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    total_records = len(results)

    normal_records = (
        results["attack_status"] == "NORMAL"
    ).sum()

    malicious_records = (
        results["attack_status"] == "MALICIOUS"
    ).sum()

    ddos_icmp = (
        results["predicted_class"] == "DDOS_ICMP"
    ).sum()

    ddos_tcp = (
        results["predicted_class"] == "DDOS_TCP"
    ).sum()

    ddos_udp = (
        results["predicted_class"] == "DDOS_UDP"
    ).sum()

    mitigation_required = malicious_records

    if mitigation_required > 0:
        mitigation_rate = (
            mitigation_required /
            total_records
        ) * 100
    else:
        mitigation_rate = 0.0

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        predicted_classes,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predicted_classes,
        labels=CLASS_NAMES
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset": "D2",
        "model": "DNN",
        "total_records": int(total_records),
        "classification_accuracy": float(accuracy),

        "normal_records": int(normal_records),
        "malicious_records": int(malicious_records),

        "ddos_icmp": int(ddos_icmp),
        "ddos_tcp": int(ddos_tcp),
        "ddos_udp": int(ddos_udp),

        "mitigation_required": int(mitigation_required),
        "mitigation_rate_percent": float(mitigation_rate),

        "mitigation_actions": {
            "DDOS_ICMP": "BLOCK_ICMP",
            "DDOS_TCP": "BLOCK_TCP",
            "DDOS_UDP": "BLOCK_UDP",
            "NORMAL_ICMP": "ALLOW_ICMP",
            "NORMAL_TCP": "ALLOW_TCP",
            "NORMAL_UDP": "ALLOW_UDP"
        },

        "classification_report": report,

        "confusion_matrix": cm.tolist(),

        "class_order": CLASS_NAMES
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("           D2 OFFLINE MITIGATION RESULTS")
    print("=" * 70)

    print(f"Total records       : {total_records}")
    print(f"Classification acc. : {accuracy:.4f}")

    print("\nClassification:")
    print(f"  Normal records    : {normal_records}")
    print(f"  Malicious records : {malicious_records}")

    print("\nDDoS attack types:")
    print(f"  DDOS_ICMP         : {ddos_icmp}")
    print(f"  DDOS_TCP          : {ddos_tcp}")
    print(f"  DDOS_UDP          : {ddos_udp}")

    print("\nMitigation:")
    print(f"  Mitigation needed : {mitigation_required}")
    print(f"  Mitigation rate   : {mitigation_rate:.2f}%")

    print("\nMitigation actions:")
    print("  DDOS_ICMP → BLOCK_ICMP")
    print("  DDOS_TCP  → BLOCK_TCP")
    print("  DDOS_UDP  → BLOCK_UDP")
    print("  NORMAL_*  → ALLOW")

    print("\nOutput files:")
    print(f"  CSV   : {OUTPUT_CSV}")
    print(f"  JSON  : {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("D2 OFFLINE MITIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()