import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "dnn",
    "dnn_dataset2_best.keras"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "dnn_dataset2_label_mapping.json"
)


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
# LOAD MODEL AND LABEL MAPPING
# ============================================================

print("\nLoading Dataset 2 DNN classifier...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    LABEL_MAPPING_PATH,
    "r"
) as file:

    label_mapping = json.load(file)

print("Model loaded successfully.")
print("Label mapping loaded successfully.")

print("\nDataset 2 class mapping:")

for class_index, class_name in label_mapping.items():

    print(
        f"{class_index} -> {class_name}"
    )


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(data):
    """
    Prepare one or more Dataset 2 traffic records
    for DNN inference.

    Dataset 2 training does not apply scaling in the
    DNN training script, so no scaler is used here.
    """

    df = data.copy()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in DATASET2_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required Dataset 2 features: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Select exact training feature order
    # --------------------------------------------------------

    X = df[
        DATASET2_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for column in DATASET2_FEATURES:

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
    # Validate missing values
    # --------------------------------------------------------

    if X.isna().any().any():

        missing_columns = (
            X.columns[
                X.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Dataset 2 input contains missing or "
            "invalid values in: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    return X.to_numpy(
        dtype=np.float32
    )


# ============================================================
# D2 CLASSIFICATION FUNCTION
# ============================================================

def classify_traffic(data):
    """
    Classify Dataset 2 traffic into one of six classes.

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

        class_name = label_mapping[
            str(predicted_class)
        ]

        confidence = float(
            probabilities[
                index,
                predicted_class
            ]
        )

        # ----------------------------------------------------
        # Store all class probabilities
        # ----------------------------------------------------

        class_probabilities = {}

        for class_index, probability in enumerate(
            probabilities[index]
        ):

            class_name_for_probability = label_mapping[
                str(class_index)
            ]

            class_probabilities[
                class_name_for_probability
            ] = float(probability)

        results.append({
            "prediction": class_name,
            "class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        })

    return results


# ============================================================
# SINGLE RECORD PREDICTION
# ============================================================

def classify_single_record(record):
    """
    Classify one Dataset 2 traffic record.

    Input:
        Dictionary containing Dataset 2 features.

    Returns:
        Prediction dictionary.
    """

    df = pd.DataFrame(
        [record]
    )

    results = classify_traffic(
        df
    )

    return results[0]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("DATASET 2 DDoS CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load real Dataset 2 test data
    # --------------------------------------------------------

    test_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset2",
        "test.csv"
    )

    test_df = pd.read_csv(
        test_path
    )

    print(
        f"\nTest dataset rows: "
        f"{len(test_df):,}"
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

    result = classify_traffic(
        sample_df
    )[0]

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPrediction Result")
    print("-" * 40)

    print(
        f"Prediction : "
        f"{result['prediction']}"
    )

    print(
        f"Class      : "
        f"{result['class']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )

    print("\nClass Probabilities:")

    for class_name, probability in (
        result["class_probabilities"].items()
    ):

        print(
            f"  {class_name:<15} : "
            f"{probability:.4f}"
        )

    print("\n" + "=" * 70)
    print("D2 CLASSIFICATION COMPLETED")
    print("=" * 70)