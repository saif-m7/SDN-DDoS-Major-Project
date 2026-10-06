import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# D2 REAL-TIME DDoS DETECTOR
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "dnn",
    "dnn_dataset2_best.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "results",
    "preprocessing",
    "dataset2_scaler.pkl"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "dnn_dataset2_label_mapping.json"
)


# ============================================================
# D2 FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
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
# D2 DETECTOR
# ============================================================

class D2Detector:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING D2 DDoS DETECTOR")
        print("=" * 70)

        # ----------------------------------------------------
        # Check required files
        # ----------------------------------------------------

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"D2 model not found:\n{MODEL_PATH}"
            )

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"D2 scaler not found:\n{SCALER_PATH}"
            )

        if not os.path.exists(LABEL_MAPPING_PATH):
            raise FileNotFoundError(
                f"D2 label mapping not found:\n{LABEL_MAPPING_PATH}"
            )

        # ----------------------------------------------------
        # Load D2 DNN model
        # ----------------------------------------------------

        print("Loading D2 DNN model...")

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        print("Loading D2 scaler...")

        self.scaler = joblib.load(
            SCALER_PATH
        )

        # ----------------------------------------------------
        # Load label mapping
        # ----------------------------------------------------

        print("Loading D2 label mapping...")

        with open(
            LABEL_MAPPING_PATH,
            "r"
        ) as file:

            self.label_mapping = json.load(file)

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print("\nD2 model loaded successfully.")
        print(f"Input features : {len(FEATURE_COLUMNS)}")
        print(f"Label mapping   : {self.label_mapping}")

        print("\nD2 Detector ready.")
        print("=" * 70)


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def prepare_features(self, record):

        # ----------------------------------------------------
        # Check required features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in record
        ]

        if missing_features:

            raise ValueError(
                "Missing D2 features: "
                + ", ".join(missing_features)
            )

        # ----------------------------------------------------
        # Create DataFrame in exact feature order
        # ----------------------------------------------------

        df = pd.DataFrame(
            [[
                record[feature]
                for feature in FEATURE_COLUMNS
            ]],
            columns=FEATURE_COLUMNS
        )

        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        for feature in FEATURE_COLUMNS:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="raise"
            )

        # ----------------------------------------------------
        # Check missing values
        # ----------------------------------------------------

        if df.isnull().any().any():

            raise ValueError(
                "D2 input contains missing values."
            )

        # ----------------------------------------------------
        # Scale using training scaler
        # ----------------------------------------------------

        X_scaled = self.scaler.transform(
            df[FEATURE_COLUMNS]
        )

        return X_scaled


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, record):

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        X_scaled = self.prepare_features(
            record
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        probabilities = self.model.predict(
            X_scaled,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        # ----------------------------------------------------
        # Convert class ID → label
        # ----------------------------------------------------

        predicted_label = self.label_mapping.get(
            str(predicted_class),
            str(predicted_class)
        )

        # ----------------------------------------------------
        # Determine normal / DDoS
        # ----------------------------------------------------

        if predicted_label.startswith("DDOS"):

            traffic_status = "MALICIOUS"

        else:

            traffic_status = "NORMAL"

        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        class_probabilities = {}

        for index, probability in enumerate(
            probabilities
        ):

            label = self.label_mapping.get(
                str(index),
                str(index)
            )

            class_probabilities[label] = float(
                probability
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {
            "class_id": predicted_class,
            "label": predicted_label,
            "status": traffic_status,
            "confidence": confidence,
            "probabilities": class_probabilities
        }

        return result


# ============================================================
# TEST FUNCTION
# ============================================================

def test_detector():

    detector = D2Detector()

    print("\n" + "=" * 70)
    print("D2 DETECTOR TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Example D2 record
    #
    # These values are only for testing the detector pipeline.
    # --------------------------------------------------------

    sample_record = {

        "total_length": 40,
        "ttl": 64,
        "proto": 6,
        "csum": 30000,
        "src_port": 1234,
        "dst_port": 80,
        "tcp_flag": 2,
        "type_icmp": 0,
        "code_icmp": 0,
        "tx_bytes_ave": 100
    }

    result = detector.predict(
        sample_record
    )

    print("\nPrediction:")
    print(
        f"Class ID   : {result['class_id']}"
    )

    print(
        f"Label      : {result['label']}"
    )

    print(
        f"Status     : {result['status']}"
    )

    print(
        f"Confidence : {result['confidence']:.4f}"
    )

    print("\nProbabilities:")

    for label, probability in result[
        "probabilities"
    ].items():

        print(
            f"  {label}: {probability:.6f}"
        )

    print("\n" + "=" * 70)
    print("D2 DETECTOR TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_detector()