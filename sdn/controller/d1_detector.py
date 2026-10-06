import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "dnn",
    "dnn_dataset1_best.keras"
)

MEDIANS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "preprocessing",
    "dataset1_medians.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "results",
    "preprocessing",
    "dataset1_scaler.pkl"
)


# ============================================================
# DATASET 1 FEATURES
# ============================================================

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
# PROTOCOL MAPPING
# ============================================================

PROTOCOL_MAPPING = {
    "ICMP": 0,
    "TCP": 1,
    "UDP": 2
}


# ============================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

print("\nLoading Dataset 1 DNN detector...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

train_medians = joblib.load(
    MEDIANS_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

print("Model loaded successfully.")
print("Preprocessing artifacts loaded successfully.")


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(data):
    """
    Preprocess one or more Dataset 1 traffic records.

    Expected input:
        pandas DataFrame

    Returns:
        Scaled NumPy array ready for DNN inference.
    """

    df = data.copy()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in DATASET1_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Dataset 1 features: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Encode Protocol
    # --------------------------------------------------------

    if df["Protocol"].dtype == object:

        df["Protocol"] = (
            df["Protocol"]
            .astype(str)
            .str.upper()
            .map(PROTOCOL_MAPPING)
        )

    # --------------------------------------------------------
    # Check protocol encoding
    # --------------------------------------------------------

    if df["Protocol"].isna().any():

        raise ValueError(
            "Unknown or invalid Protocol value found. "
            "Expected ICMP, TCP, or UDP."
        )

    # --------------------------------------------------------
    # Select exact training feature order
    # --------------------------------------------------------

    X = df[DATASET1_FEATURES].copy()

    # --------------------------------------------------------
    # Convert all features to numeric
    # --------------------------------------------------------

    for column in DATASET1_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for column in DATASET1_FEATURES:

        if X[column].isna().any():

            X[column] = X[column].fillna(
                train_medians[column]
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X.isna().any().any():

        raise ValueError(
            "Input still contains missing values "
            "after median imputation."
        )

    # --------------------------------------------------------
    # Apply training scaler
    #
    # IMPORTANT:
    # Keep X as a pandas DataFrame here so that the
    # StandardScaler receives the same feature names
    # used during training.
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Convert scaled data to float32 for TensorFlow
    # --------------------------------------------------------

    return X_scaled.astype(
        np.float32
    )


# ============================================================
# D1 DETECTION FUNCTION
# ============================================================

def detect_ddos(data):
    """
    Detect normal or malicious traffic using
    the trained Dataset 1 DNN.

    Returns:
        List of prediction dictionaries.
    """

    X = preprocess_input(
        data
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Prepare results
    # --------------------------------------------------------

    results = []

    for index in range(
        len(predictions)
    ):

        predicted_class = int(
            predictions[index]
        )

        normal_probability = float(
            probabilities[index][0]
        )

        malicious_probability = float(
            probabilities[index][1]
        )

        if predicted_class == 1:

            label = "MALICIOUS"

        else:

            label = "NORMAL"

        results.append({
            "prediction": label,
            "class": predicted_class,
            "normal_probability": normal_probability,
            "malicious_probability": malicious_probability
        })

    return results


# ============================================================
# SINGLE RECORD PREDICTION
# ============================================================

def detect_single_record(record):
    """
    Detect one Dataset 1 traffic record.

    Input:
        Dictionary containing Dataset 1 features.

    Returns:
        Prediction dictionary.
    """

    df = pd.DataFrame(
        [record]
    )

    results = detect_ddos(
        df
    )

    return results[0]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("DATASET 1 DDoS DETECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load one real test sample
    # --------------------------------------------------------

    test_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset1",
        "test.csv"
    )

    test_df = pd.read_csv(
        test_path
    )

    # --------------------------------------------------------
    # Select first test record
    # --------------------------------------------------------

    sample = test_df.iloc[0]

    sample_df = pd.DataFrame(
        [sample]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = detect_ddos(
        sample_df
    )[0]

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPrediction Result")
    print("-" * 40)

    print(
        f"Prediction            : "
        f"{result['prediction']}"
    )

    print(
        f"Class                 : "
        f"{result['class']}"
    )

    print(
        f"Normal Probability    : "
        f"{result['normal_probability']:.4f}"
    )

    print(
        f"Malicious Probability : "
        f"{result['malicious_probability']:.4f}"
    )

    print("\n" + "=" * 70)
    print("D1 DETECTION COMPLETED")
    print("=" * 70)